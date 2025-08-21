import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
from utils import vn_tags
import editdistance
from difflib import SequenceMatcher
from torch.cuda.amp import autocast


def validation(model, criterion, evaluation_loader, converter):
    """ validation or evaluation """

    norm_ED = 0
    norm_ED_wer = 0

    tot_ED = 0
    tot_ED_wer = 0

    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    # Extra metrics for composed Vietnamese output
    norm_ED_final = 0
    tot_ED_final = 0
    length_of_gt_final = 0
    ter_num, ter_den = 0, 0
    mer_num, mer_den = 0, 0
    count = 0
    all_preds_str = []
    all_labels = []

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.cuda()

        enc = converter.encode(labels)
        # Support both converters: take base tensors for CTC
        if isinstance(enc, tuple) and len(enc) >= 2:
            text_for_loss, length_for_loss = enc[0], enc[1]
        else:
            text_for_loss, length_for_loss = enc
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        with autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            outs = model(image)
        use_dual = isinstance(outs, dict)
        base_logits = outs['base'] if use_dual else outs
        base_logits = base_logits.float()
        preds_size = torch.IntTensor(
            [base_logits.size(1)] * batch_size).to(base_logits.device)
        base_logp = base_logits.permute(1, 0, 2).log_softmax(2)

        torch.backends.cudnn.enabled = False
        cost = criterion(base_logp, text_for_loss.to(
            base_logits.device), preds_size, length_for_loss.to(base_logits.device)).mean()
        torch.backends.cudnn.enabled = True

        _, preds_index = base_logp.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index, preds_size)
        valid_loss += cost.item()
        count += 1

        all_preds_str.extend(preds_str)
        all_labels.extend(labels)

        for pred_cer, gt_cer in zip(preds_str, labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                norm_ED += tmp_ED / float(len(gt_cer))
            tot_ED += tmp_ED
            length_of_gt += len(gt_cer)

        for pred_wer, gt_wer in zip(preds_str, labels):
            pred_wer = utils.format_string_for_wer(pred_wer)
            gt_wer = utils.format_string_for_wer(gt_wer)
            pred_wer = pred_wer.split(" ")
            gt_wer = gt_wer.split(" ")
            tmp_ED_wer = editdistance.eval(pred_wer, gt_wer)

            if len(gt_wer) == 0:
                norm_ED_wer += 1
            else:
                norm_ED_wer += tmp_ED_wer / float(len(gt_wer))

            tot_ED_wer += tmp_ED_wer
            length_of_gt_wer += len(gt_wer)

        # If dual-head, compose Vietnamese output and compute extra metrics
        if use_dual:
            # Greedy best-path segments from base
            base_argmax = base_logits.argmax(dim=2)  # [B, T]
            mod_sm = torch.softmax(outs['mod'], dim=2)
            tone_sm = torch.softmax(outs['tone'], dim=2)
            composed_batch = []
            for b in range(batch_size):
                # Decode base best-path string (already in preds_str), also get best-path segments
                frames = base_argmax[b].detach().cpu().tolist()
                segments = []
                prev = None
                current_frames = []
                for t, k in enumerate(frames):
                    if k == 0:
                        if prev is not None:
                            segments.append((prev, current_frames))
                            prev = None
                            current_frames = []
                        continue
                    if prev is None or k != prev:
                        if prev is not None and current_frames:
                            segments.append((prev, current_frames))
                        prev = int(k)
                        current_frames = [t]
                    else:
                        current_frames.append(t)
                if prev is not None and current_frames:
                    segments.append((prev, current_frames))

                # Reconstruct base string from segments via converter.character
                # converter.character[cls_idx] gives base char for cls_idx (1-based)
                base_chars = []
                mod_ids = []
                tone_ids = []
                for cls_idx, t_idx in segments:
                    ch = converter.character[cls_idx] if cls_idx < len(
                        converter.character) else ''
                    if ch:
                        base_chars.append(ch)
                        pm = mod_sm[b, t_idx, :].mean(dim=0)
                        pt = tone_sm[b, t_idx, :].mean(dim=0)
                        mod_ids.append(int(pm.argmax().item()))
                        tone_ids.append(int(pt.argmax().item()))
                composed = vn_tags.compose_str(
                    ''.join(base_chars), mod_ids, tone_ids)
                composed_batch.append(composed)
                # Compute CER on composed vs ground-truth label
                gt = labels[b]
                tmp_ED_c = editdistance.eval(composed, gt)
                if len(gt) == 0:
                    norm_ED_final += 1
                else:
                    norm_ED_final += tmp_ED_c / float(len(gt))
                tot_ED_final += tmp_ED_c
                length_of_gt_final += len(gt)

                # Tone / Modifier error rates with base-aligned spans using SequenceMatcher
                base_gt, mod_gt, tone_gt = vn_tags.decompose_str(gt)
                base_pr, mod_pr, tone_pr = vn_tags.decompose_str(composed)
                sm = SequenceMatcher(None, base_gt, base_pr)
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op != 'equal':
                        continue
                    for k in range(i2 - i1):
                        ch = base_gt[i1 + k]
                        if vn_tags.is_vowel(ch):
                            ter_den += 1
                            if tone_gt[i1 + k] != tone_pr[j1 + k]:
                                ter_num += 1
                            # Also compute modifier error rate on vowels
                            mer_den += 1
                            if mod_gt[i1 + k] != mod_pr[j1 + k]:
                                mer_num += 1

                # Replace the just-computed base pred with composed for external reporting if needed
                all_preds_str[-batch_size + b] = composed
            # Overwrite current batch predictions with composed VN strings so return value is the final output
            preds_str = composed_batch

    # If dual-head was used at least once, compute final WER on composed outputs
    wer_tot_final, wer_len_final = 0, 0
    if 'composed_batch' in locals() and len(composed_batch) > 0:
        for comp, gt in zip(composed_batch, labels):
            p = utils.format_string_for_wer(comp).split(" ")
            g = utils.format_string_for_wer(gt).split(" ")
            wer_tot_final += editdistance.eval(p, g)
            wer_len_final += len(g)

    # at the end, compute both sets
    val_loss = valid_loss / count
    CER_base = tot_ED / float(length_of_gt) if length_of_gt > 0 else 0.0
    WER_base = tot_ED_wer / \
        float(length_of_gt_wer) if length_of_gt_wer > 0 else 0.0

    if 'outs' in locals() and isinstance(outs, dict):
        CER_final = tot_ED_final / \
            float(length_of_gt_final) if length_of_gt_final > 0 else 0.0
        WER_final = wer_tot_final / \
            float(wer_len_final) if wer_len_final > 0 else 0.0
        TER = ter_num / float(ter_den) if ter_den > 0 else 0.0
        MER = mer_num / float(mer_den) if mer_den > 0 else 0.0
        print(
            f"[VALID] Composed CER (VN): {CER_final:.4f} | WER: {WER_final:.4f} | TER: {TER:.4f} ({ter_num}/{ter_den}) | MER: {MER:.4f}")
        return val_loss, CER_final, WER_final, composed_batch, labels

    # non-dual fallback
    return val_loss, CER_base, WER_base, preds_str, labels
