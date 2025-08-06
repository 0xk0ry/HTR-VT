import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np

from utils import utils
from data.dataset import merge_base_and_diacritic
import editdistance


def validation(model, criterion, evaluation_loader, converter):
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
    all_diacritic_acc = []

    for i, (image_tensors, base_labels, diacritic_labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.cuda()

        # Encode base labels for CTC loss
        text_for_loss, length_for_loss = converter.encode(base_labels)
        
        # Get model outputs (both base and diacritic)
        base_logits, diacritic_logits = model(image)
        base_logits = base_logits.float()
        preds_size = torch.IntTensor([base_logits.size(1)] * batch_size)
        base_logits_ctc = base_logits.permute(1, 0, 2).log_softmax(2)

        torch.backends.cudnn.enabled = False
        cost = criterion(base_logits_ctc, text_for_loss, preds_size, length_for_loss).mean()
        torch.backends.cudnn.enabled = True

        # Decode base character predictions
        _, preds_index = base_logits_ctc.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        base_preds_str = converter.decode(preds_index.data, preds_size.data)
        
        # Decode diacritic predictions
        diacritic_pred = diacritic_logits.argmax(dim=-1).cpu().numpy()  # (B, T)
        
        # Merge base and diacritic predictions to get final Vietnamese text
        merged_preds = []
        for b_idx in range(batch_size):
            base_str = base_preds_str[b_idx]
            diac_seq = diacritic_pred[b_idx][:len(base_str)]  # Match length
            merged_text = merge_base_and_diacritic(base_str, diac_seq)
            merged_preds.append(merged_text)
        
        # Compute diacritic accuracy
        for b_idx in range(batch_size):
            pred_diac = diacritic_pred[b_idx]
            true_diac = diacritic_labels[b_idx]
            # Pad to same length for comparison
            min_len = min(len(pred_diac), len(true_diac))
            if min_len > 0:
                acc = np.mean(np.array(pred_diac[:min_len]) == np.array(true_diac[:min_len]))
                all_diacritic_acc.append(acc)
        
        valid_loss += cost.item()
        count += 1

        all_preds_str.extend(merged_preds)
        # Use original labels for CER/WER computation
        original_labels = [merge_base_and_diacritic(base_labels[i], diacritic_labels[i]) 
                          for i in range(batch_size)]
        all_labels.extend(original_labels)

        # Compute CER on merged predictions vs original labels
        for pred_cer, gt_cer in zip(merged_preds, original_labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                norm_ED += tmp_ED / float(len(gt_cer))
            tot_ED += tmp_ED
            length_of_gt += len(gt_cer)

        # Compute WER on merged predictions vs original labels
        for pred_wer, gt_wer in zip(merged_preds, original_labels):
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
    avg_diacritic_acc = np.mean(all_diacritic_acc) if all_diacritic_acc else 0.0

    return val_loss, CER, WER, all_preds_str, all_labels, avg_diacritic_acc