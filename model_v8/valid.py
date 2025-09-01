import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance
from difflib import SequenceMatcher


def validation(model, criterion, evaluation_loader, converter, args=None):
    """Validation with extended metrics.

    Returns:
        val_loss_base: base CTC validation loss (no tone component)
        val_loss_full: base + tone validation loss (mirrors training objective)
        CER_base: CER on base-only (no tone) strings
        WER_base: WER on base-only strings
        CER: CER on full composed (with tone) strings
        WER: WER on full composed strings
        TER: Tone Error Rate (tone symbol mismatch rate per character position)
        all_preds_str: list of full composed predictions
        all_labels: list of ground-truth full strings
    """

    # Full (with tone) metrics accumulators
    tot_ED = 0            # char edit distance full
    tot_ED_wer = 0        # word edit distance full

    valid_loss_base = 0.0      # base CTC loss only
    valid_loss_full = 0.0      # base + tone (mirrors training)
    length_of_gt = 0      # total chars full
    length_of_gt_wer = 0  # total words full
    count = 0
    all_preds_str = []
    all_labels = []

    # Base-only metrics (strip tones)
    base_tot_ED = 0
    base_length_gt = 0
    base_tot_ED_wer = 0
    base_length_gt_wer = 0

    # Tone metrics
    tone_errors = 0
    tone_total = 0

    use_tone = hasattr(model, 'tone_head') or isinstance(getattr(model, 'head', None), torch.nn.Linear)
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        image = image_tensors.cuda()
        batch_size = image.size(0)
        text_base, length_base, tone_lists, _ = converter.encode(labels)
        outputs = model(image)
        if isinstance(outputs, dict):
            base_logits = outputs['base']
            tone_logits = outputs['tone']
        else:
            base_logits = outputs
            tone_logits = None

        base_logits = base_logits.float()
        device = base_logits.device
        preds_size = torch.full((batch_size,), base_logits.size(1), dtype=torch.int32, device=device)
        text_base = text_base.to(device, non_blocking=True)
        length_base = length_base.to(device, non_blocking=True)
        base_logp = base_logits.permute(1, 0, 2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        base_ctc = criterion(base_logp, text_base, preds_size, length_base).mean()
        torch.backends.cudnn.enabled = True
        _, base_idx = base_logp.max(2)
        base_idx_flat = base_idx.transpose(1, 0).contiguous().view(-1)
        base_strs = converter.decode(base_idx_flat.data, preds_size.data)

        preds_str = base_strs
        pred_tone_sequences = []
        if tone_logits is not None:
            # Pool in probability space then take log (geometric mean) to avoid over-penalizing variance
            tone_prob_frames = torch.softmax(tone_logits, dim=2)
            constrained_tone_ids_flat = []
            for b in range(batch_size):
                frame_base = base_logits[b].argmax(dim=1)
                frame_tone_logp = tone_prob_frames[b]
                prev = -1
                seg_base_ids = []
                seg_tone_logp = []
                for t in range(frame_base.size(0)):
                    cls_id = frame_base[t].item()
                    if cls_id == 0:
                        continue
                    if cls_id != prev:
                        prev = cls_id
                        seg_base_ids.append(cls_id)
                        seg_tone_logp.append([])
                    seg_tone_logp[-1].append(frame_tone_logp[t])  # collect probability vectors
                if not seg_base_ids:
                    pred_tone_sequences.append([])
                    continue
                # Average probabilities across frames for each collapsed segment, then convert to log for scoring
                seg_tone_logp = [torch.log((torch.stack(v, dim=0).mean(dim=0) if len(v) > 1 else v[0]) + 1e-8) for v in seg_tone_logp]
                base_charset = converter.character[1:]
                collapsed_chars = []
                for cid in seg_base_ids:
                    if 1 <= cid <= len(base_charset):
                        collapsed_chars.append(base_charset[cid-1])
                    else:
                        collapsed_chars.append('')
                C_tone = seg_tone_logp[0].numel() if seg_tone_logp else 6
                level_idx = 0
                vowel_fn = converter.vn_tags.is_vowel
                tone_ids = [0] * len(seg_base_ids)
                level_scores = [vec[level_idx].item() for vec in seg_tone_logp]
                i_char = 0
                while i_char < len(seg_base_ids):
                    start = i_char
                    while i_char < len(seg_base_ids) and collapsed_chars[i_char] != ' ':
                        i_char += 1
                    end = i_char
                    syll_indices = list(range(start, end))
                    if syll_indices:
                        score_none = sum(level_scores[j] for j in syll_indices)
                        best_score = score_none
                        best_assign = None
                        for j in syll_indices:
                            ch = collapsed_chars[j]
                            if not vowel_fn(ch):
                                continue
                            logp_vec = seg_tone_logp[j]
                            if C_tone > 1:
                                non_level_vals = logp_vec[1:]
                                best_val, best_local = torch.max(non_level_vals, dim=0)
                                tone_cls = best_local.item() + 1
                                score_j = best_val.item() + sum(level_scores[k] for k in syll_indices if k != j)
                                if score_j > best_score:
                                    best_score = score_j
                                    best_assign = (j, tone_cls)
                        if best_assign is not None:
                            j_sel, tone_cls_sel = best_assign
                            tone_ids[j_sel] = tone_cls_sel
                    if i_char < len(seg_base_ids) and collapsed_chars[i_char] == ' ':
                        i_char += 1
                pred_tone_sequences.append(tone_ids)
                constrained_tone_ids_flat.extend(tone_ids)
            tone_char_preds_tensor = torch.tensor(constrained_tone_ids_flat, device=base_logits.device)
            try:
                preds_str = converter.decode_with_tones(base_idx_flat.data, preds_size.data, tone_char_preds_tensor)
            except Exception:
                preds_str = base_strs
        else:
            pred_tone_sequences = [[] for _ in range(batch_size)]
        # Tone-inclusive loss (probability pooling); recompute minimal alignment under no_grad
        lambda_tone = getattr(args, 'lambda_tone', 1.0) if args is not None else 1.0
        full_loss = base_ctc.item()
        if tone_logits is not None and lambda_tone != 0:
            with torch.no_grad():
                frame_argmax = base_logits.argmax(dim=2)
                p_tone_frames = torch.softmax(tone_logits, dim=2)
                base_chars_full = ''.join(converter.character[1:])
                vowel_fn = converter.vn_tags.is_vowel
                alpha_consonant = 0.2
                offset = 0
                batch_losses = []
                for b in range(batch_size):
                    U = int(length_base[b].item())
                    if U == 0:
                        continue
                    target_seq = text_base[offset:offset+U].tolist()
                    tone_target = tone_lists[b].to(device)
                    offset += U
                    frames = frame_argmax[b]
                    non_blank = frames.ne(0)
                    prevf = torch.cat([frames.new_zeros(1), frames[:-1]])
                    change = frames.ne(prevf) & non_blank
                    starts = torch.nonzero(change, as_tuple=True)[0]
                    if starts.numel() == 0:
                        continue
                    seg_vals = frames[starts]
                    segments = []
                    for si in range(starts.numel()):
                        st = starts[si]
                        en = starts[si+1] if si+1 < starts.numel() else frames.numel()
                        segments.append((int(seg_vals[si].item()), torch.arange(st, en, device=device)))
                    seg_ptr = 0
                    char_losses = []
                    vowel_mask = []
                    for u, cls in enumerate(target_seq):
                        while seg_ptr < len(segments) and segments[seg_ptr][0] != cls:
                            seg_ptr += 1
                        if seg_ptr >= len(segments):
                            continue
                        _, idxs = segments[seg_ptr]
                        seg_ptr += 1
                        seg_p = p_tone_frames[b, idxs, :].mean(dim=0)
                        tone_id = tone_target[u]
                        char_losses.append(-torch.log(seg_p[tone_id] + 1e-8))
                        ch = base_chars_full[cls-1] if 1 <= cls <= len(base_chars_full) else ''
                        vowel_mask.append(1.0 if vowel_fn(ch) else 0.0)
                    if char_losses:
                        lc = torch.stack(char_losses)
                        vm = torch.tensor(vowel_mask, device=device)
                        w = vm + alpha_consonant * (1.0 - vm)
                        batch_losses.append((lc * w).sum() / (w.sum() + 1e-6))
                if batch_losses:
                    tone_loss_batch = torch.stack(batch_losses).mean().item()
                    full_loss += lambda_tone * tone_loss_batch
        valid_loss_full += full_loss
        valid_loss_base += base_ctc.item()
        count += 1

        all_preds_str.extend(preds_str)
        all_labels.extend(labels)
        for pred_full, gt_full, pred_base, pred_tones in zip(preds_str, labels, base_strs, pred_tone_sequences):
            ed_full = editdistance.eval(pred_full, gt_full)
            tot_ED += ed_full
            length_of_gt += len(gt_full)
            pred_full_fmt = utils.format_string_for_wer(pred_full).split(' ')
            gt_full_fmt = utils.format_string_for_wer(gt_full).split(' ')
            ed_full_w = editdistance.eval(pred_full_fmt, gt_full_fmt)
            tot_ED_wer += ed_full_w
            length_of_gt_wer += len(gt_full_fmt)
            gt_base, gt_tones = converter.vn_tags.decompose_str(gt_full)
            ed_base = editdistance.eval(pred_base, gt_base)
            base_tot_ED += ed_base
            base_length_gt += len(gt_base)
            pred_base_fmt = utils.format_string_for_wer(pred_base).split(' ')
            gt_base_fmt = utils.format_string_for_wer(gt_base).split(' ')
            ed_base_w = editdistance.eval(pred_base_fmt, gt_base_fmt)
            base_tot_ED_wer += ed_base_w
            base_length_gt_wer += len(gt_base_fmt)
            sm = SequenceMatcher(None, gt_base, pred_base)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != 'equal':
                    continue
                span_len = i2 - i1
                if span_len <= 0:
                    continue
                tone_total += span_len
                for k in range(span_len):
                    if j1 + k >= len(pred_tones) or i1 + k >= len(gt_tones):
                        tone_errors += 1
                    elif gt_tones[i1 + k] != pred_tones[j1 + k]:
                        tone_errors += 1

    val_loss_base = valid_loss_base / count
    val_loss_full = valid_loss_full / count
    CER_full = tot_ED / float(length_of_gt) if length_of_gt > 0 else 0.0
    WER_full = tot_ED_wer / float(length_of_gt_wer) if length_of_gt_wer > 0 else 0.0
    CER_base = base_tot_ED / float(base_length_gt) if base_length_gt > 0 else 0.0
    WER_base = base_tot_ED_wer / float(base_length_gt_wer) if base_length_gt_wer > 0 else 0.0
    TER = tone_errors / float(tone_total) if tone_total > 0 else 0.0

    return val_loss_base, val_loss_full, CER_base, WER_base, CER_full, WER_full, TER, all_preds_str, all_labels
