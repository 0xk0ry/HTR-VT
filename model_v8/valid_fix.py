import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance


def validation(model, criterion, evaluation_loader, converter):
    """Validation supporting tone head (base CTC + tone decode)."""

    norm_ED = 0
    norm_ED_wer = 0

    tot_ED = 0
    tot_ED_wer = 0

    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    count = 0
    all_preds_str = []
    all_labels = []

    use_tone = hasattr(model, 'tone_head') or isinstance(getattr(model, 'head', None), torch.nn.Linear)
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
        base_logp = base_logits.permute(1,0,2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        cost = criterion(base_logp, text_base, preds_size, length_base).mean()
        torch.backends.cudnn.enabled = True

        # Greedy indices (T,B)
        _, base_idx = base_logp.max(2)
        # Move to CPU for decoder consistency & reshape to (B,T)
        base_idx_bt = base_idx.permute(1,0).contiguous().cpu()

        # Original decode (may yield empty strings if all blanks)
        try:
            base_strs = converter.decode(base_idx_bt.view(-1), preds_size)
        except Exception:
            # Fallback: if converter expects (B,T) directly
            try:
                base_strs = converter.decode(base_idx_bt, preds_size)
            except Exception:
                base_strs = []

        # Manual greedy CTC collapse fallback if empty predictions
        if not base_strs or all(s == '' for s in base_strs):
            # Attempt to access charset (adapt attribute names if different)
            charset = getattr(converter, 'characters', None) or getattr(converter, 'character', None)
            blank_idx = 0
            fallback = []
            for b in range(batch_size):
                seq = base_idx_bt[b].tolist()
                out = []
                prev = None
                for idx in seq:
                    if idx != blank_idx and idx != prev:
                        if charset and idx-1 < len(charset) and idx > 0:
                            out.append(charset[idx-1])  # shift if blank at 0
                        else:
                            out.append(str(idx))
                    prev = idx
                fallback.append(''.join(out))
            base_strs = fallback

        # Tone decode (keep existing but use base_logits/base_idx_bt)
        if tone_logits is not None:
            tone_frame = tone_logits.argmax(dim=2)  # (B,T')
            tone_char_preds = []
            for b in range(batch_size):
                fcls = base_logits[b].argmax(dim=1)  # (T')
                tcls = tone_frame[b]
                prev = -1
                tc = []
                for fc, tcid in zip(fcls.tolist(), tcls.tolist()):
                    if fc != 0 and fc != prev:
                        tc.append(tcid)
                        prev = fc
                tone_char_preds.extend(tc)
            tone_char_preds_tensor = torch.tensor(tone_char_preds, device=base_logits.device)
            try:
                preds_str = converter.decode_with_tones(
                    base_idx_bt.view(-1), preds_size, tone_char_preds_tensor)
            except Exception:
                # Fallback: if tone decode fails, use base_strs
                preds_str = base_strs
        else:
            preds_str = base_strs

        valid_loss += cost.item()
        count += 1

        # Debug stats if still empty
        if all(p == '' for p in preds_str):
            with torch.no_grad():
                blanks = (base_idx_bt == 0).float().mean().item()
            print(f'[DEBUG] All predictions empty. Blank ratio={blanks:.3f} T={base_idx_bt.size(1)}')

        for (label, pred) in zip(labels, preds_str):
            print(f'Label: {label}, Pred: {pred}')
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

    val_loss = valid_loss / count
    CER = tot_ED / float(length_of_gt)
    WER = tot_ED_wer / float(length_of_gt_wer)

    return val_loss, CER, WER, all_preds_str, all_labels