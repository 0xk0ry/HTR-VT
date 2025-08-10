import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils import utils
import editdistance


def validation(model, criterion, evaluation_loader, converter, args=None):
    """ validation or evaluation """

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

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.cuda()

        text_for_loss, length_for_loss = converter.encode(labels)

        outputs = model(image)
        if isinstance(outputs, (list, tuple)):
            base_logits, tone_logits = outputs
        else:
            base_logits, tone_logits = outputs, None
        base_logits = base_logits.float()
        preds_size = torch.IntTensor([base_logits.size(1)] * batch_size)
        log_probs_tbc = base_logits.permute(1, 0, 2).log_softmax(2)

        torch.backends.cudnn.enabled = False
        cost = criterion(log_probs_tbc, text_for_loss, preds_size, length_for_loss).mean()
        torch.backends.cudnn.enabled = True

        # Greedy CTC decode (base) for baseline string
        _, preds_index = log_probs_tbc.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str_base = converter.decode(preds_index.data, preds_size.data)

        # Compose tones using tone head if available
        preds_str = []
        if tone_logits is None or args is None:
            preds_str = preds_str_base
        else:
            base_probs = F.softmax(base_logits, dim=-1)  # (B, T, C)
            tone_probs = F.softmax(tone_logits, dim=-1)  # (B, T, 6)
            # Vowel mask and gating threshold
            vowel_idxs = utils.vowel_indices_from_converter(converter)
            C = base_probs.size(-1)
            vowel_mask = torch.zeros(C, device=base_probs.device)
            if len(vowel_idxs) > 0:
                vowel_mask.scatter_(0, torch.tensor(vowel_idxs, device=base_probs.device, dtype=torch.long), 1.0)
            tau_v = getattr(args, 'tau_v', 0.5)
            kappa = getattr(args, 'tone_kappa', 0.3)

            # Framewise argmax labels
            frame_labels = base_logits.argmax(dim=-1)  # (B, T)

            for b in range(batch_size):
                T = base_logits.size(1)
                spans = []  # list of (c_idx, start, end) inclusive of start, exclusive of end
                last_c = 0
                start = None
                for t in range(T):
                    c = int(frame_labels[b, t].item())
                    if c == 0:  # blank
                        if start is not None:
                            spans.append((last_c, start, t))
                            start = None
                        last_c = 0
                        continue
                    if start is None:
                        start = t
                        last_c = c
                    elif c != last_c:
                        spans.append((last_c, start, t))
                        start = t
                        last_c = c
                if start is not None:
                    spans.append((last_c, start, T))

                # Build composed string from spans
                chars = []
                for c_idx, s, e in spans:
                    ch = converter.character[c_idx] if c_idx < len(converter.character) else ''
                    # Skip safety
                    if ch == '' or c_idx == 0:
                        continue
                    if utils.is_vietnamese_vowel(ch) and tone_logits is not None:
                        # Gate per frame in span
                        v_scores = (base_probs[b, s:e, :] * vowel_mask.view(1, -1)).sum(dim=-1)  # (L,)
                        gate = (v_scores >= tau_v).float()
                        denom = gate.sum()
                        if denom.item() < 1e-6:
                            gate = torch.ones_like(gate)
                            denom = gate.sum()
                        avg_probs = (tone_probs[b, s:e, :] * gate.view(-1, 1)).sum(dim=0) / denom
                        # Choose tone vs NONE by margin
                        best_tone = int(avg_probs[1:].argmax().item()) + 1  # 1..5
                        margin = float(avg_probs[best_tone] - avg_probs[0])
                        tone_id = best_tone if margin >= kappa else 0
                        ch = utils.apply_tone_to_char(ch, tone_id)
                    chars.append(ch)

                preds_str.append(''.join(chars))

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

    val_loss = valid_loss / count
    CER = tot_ED / float(length_of_gt)
    WER = tot_ED_wer / float(length_of_gt_wer)

    return val_loss, CER, WER, all_preds_str, all_labels
