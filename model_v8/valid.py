import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance


def validation(model, criterion, evaluation_loader, converter):
    """Validation with extended metrics.

    Returns:
        val_loss: base CTC validation loss (no tone component)
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

    valid_loss = 0.0      # base CTC loss only
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

    use_tone = hasattr(model, 'tone_head') or isinstance(
        getattr(model, 'head', None), torch.nn.Linear)
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        image = image_tensors.cuda()
        batch_size = image.size(0)
        enc = converter.encode(labels)
        text_base, length_base, tone_lists, _ = enc
        outputs = model(image)
        if isinstance(outputs, dict):
            base_logits = outputs['base']
            tone_logits = outputs['tone']
        else:
            base_logits = outputs
            tone_logits = None
        base_logits = base_logits.float()
        preds_size = torch.IntTensor([base_logits.size(1)] * batch_size)
        base_logp = base_logits.permute(1, 0, 2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        cost = criterion(base_logp, text_base, preds_size, length_base).mean()
        torch.backends.cudnn.enabled = True
        _, base_idx = base_logp.max(2)              # (T,B)
        base_idx_flat = base_idx.transpose(1, 0).contiguous().view(-1)
        base_strs = converter.decode(base_idx_flat.data, preds_size.data)

        preds_str = base_strs  # default (no tone)
        # per-sample predicted tone ids aligned to collapsed base chars
        pred_tone_sequences = []
        if tone_logits is not None:
            tone_frame = tone_logits.argmax(dim=2)  # (B,T)
            tone_char_preds_flat = []
            for b in range(batch_size):
                fcls = base_logits[b].argmax(dim=1)  # (T)
                tcls = tone_frame[b]
                prev = -1
                tc = []
                for fc, tcid in zip(fcls.tolist(), tcls.tolist()):
                    if fc != 0 and fc != prev:
                        tc.append(tcid)
                        prev = fc
                pred_tone_sequences.append(tc)
                tone_char_preds_flat.extend(tc)
            tone_char_preds_tensor = torch.tensor(
                tone_char_preds_flat, device=base_logits.device)
            try:
                preds_str = converter.decode_with_tones(
                    base_idx_flat.data, preds_size.data, tone_char_preds_tensor)
            except Exception:
                preds_str = base_strs
        else:
            # Maintain length for downstream loops
            pred_tone_sequences = [[] for _ in range(batch_size)]
        valid_loss += cost.item()
        count += 1

        all_preds_str.extend(preds_str)
        all_labels.extend(labels)
        # Per-sample metrics
        for idx_sample, (pred_full, gt_full, pred_base, pred_tones) in enumerate(zip(preds_str, labels, base_strs, pred_tone_sequences)):
            # Full CER
            ed_full = editdistance.eval(pred_full, gt_full)
            tot_ED += ed_full
            length_of_gt += len(gt_full)
            # Full WER
            pred_full_fmt = utils.format_string_for_wer(pred_full).split(" ")
            gt_full_fmt = utils.format_string_for_wer(gt_full).split(" ")
            ed_full_w = editdistance.eval(pred_full_fmt, gt_full_fmt)
            tot_ED_wer += ed_full_w
            length_of_gt_wer += len(gt_full_fmt)

            # Base forms
            gt_base, gt_tones = converter.vn_tags.decompose_str(gt_full)
            ed_base = editdistance.eval(pred_base, gt_base)
            base_tot_ED += ed_base
            base_length_gt += len(gt_base)
            pred_base_fmt = utils.format_string_for_wer(pred_base).split(" ")
            gt_base_fmt = utils.format_string_for_wer(gt_base).split(" ")
            ed_base_w = editdistance.eval(pred_base_fmt, gt_base_fmt)
            base_tot_ED_wer += ed_base_w
            base_length_gt_wer += len(gt_base_fmt)

            # Tone error rate (per character position)
            tone_total += len(gt_tones)
            # Count mismatches including missing predictions beyond length
            for pos in range(len(gt_tones)):
                if pos >= len(pred_tones) or gt_tones[pos] != pred_tones[pos]:
                    tone_errors += 1

    val_loss = valid_loss / count
    CER_full = tot_ED / float(length_of_gt) if length_of_gt > 0 else 0.0
    WER_full = tot_ED_wer / \
        float(length_of_gt_wer) if length_of_gt_wer > 0 else 0.0
    CER_base = base_tot_ED / \
        float(base_length_gt) if base_length_gt > 0 else 0.0
    WER_base = base_tot_ED_wer / \
        float(base_length_gt_wer) if base_length_gt_wer > 0 else 0.0
    TER = tone_errors / float(tone_total) if tone_total > 0 else 0.0

    return val_loss, CER_base, WER_base, CER_full, WER_full, TER, all_preds_str, all_labels
