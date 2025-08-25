import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance


def validation(model, criterion, evaluation_loader, converter):
    """ validation or evaluation
    Supports both single-head (CTC) and tone-head modes.
    """

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

    use_tone_head = getattr(model, 'use_tone_head', False)

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.cuda()

        if use_tone_head and isinstance(converter, utils.ToneLabelConverter):
            enc = converter.encode(labels)
            text_base, length_base, tags_tone_list, per_sample_U = enc
            outputs = model(image)
            base_logits = outputs['base']  # [B, T, C]
            tone_logits = outputs['tone']  # [B, T, 6]

            # CTC loss on base
            preds_size = torch.IntTensor([base_logits.size(1)] * batch_size)
            base_logp = base_logits.permute(1, 0, 2).log_softmax(2)
            torch.backends.cudnn.enabled = False
            cost = criterion(base_logp, text_base, preds_size, length_base).mean()
            torch.backends.cudnn.enabled = True

            # Decode base predictions
            _, base_idx = base_logp.max(2)
            base_idx = base_idx.transpose(1, 0).contiguous().view(-1)
            base_strs = converter.decode(base_idx.data, preds_size.data)

            # Greedy tone decoding per frame -> per char by simple argmax alignment
            tone_pred = tone_logits.argmax(dim=2)  # [B, T]
            # Collapse tone predictions using same alignment as base collapse
            tone_char_preds = []
            for b in range(batch_size):
                frames = base_logits[b].argmax(dim=1)  # [T]
                tone_frames = tone_pred[b]            # [T]
                prev = None
                collapsed = []
                tone_collapsed = []
                for f, t_id in zip(frames.tolist(), tone_frames.tolist()):
                    if f != 0 and f != prev:
                        collapsed.append(f)
                        tone_collapsed.append(t_id)
                        prev = f
                tone_char_preds.extend(tone_collapsed)

            tone_char_preds_tensor = torch.tensor(tone_char_preds, device=base_logits.device)
            preds_str = converter.decode_with_tones(base_idx.data, preds_size.data, tone_char_preds_tensor)
        else:
            text_for_loss, length_for_loss = converter.encode(labels)
            preds = model(image)
            preds = preds.float()
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2).log_softmax(2)

            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()
            torch.backends.cudnn.enabled = True

            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

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

    return val_loss, CER, WER, preds_str, labels