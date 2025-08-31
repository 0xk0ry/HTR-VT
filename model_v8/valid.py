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
        _, base_idx = base_logp.max(2)
        base_idx = base_idx.transpose(1,0).contiguous().view(-1)
        base_strs = converter.decode(base_idx.data, preds_size.data)
        if tone_logits is not None:
            tone_frame = tone_logits.argmax(dim=2)
            # Align tone like training: collapse same frame classes
            tone_char_preds = []
            for b in range(batch_size):
                fcls = base_logits[b].argmax(dim=1)
                tcls = tone_frame[b]
                prev=-1
                collapsed=[]
                tc=[]
                for fc, tcid in zip(fcls.tolist(), tcls.tolist()):
                    if fc!=0 and fc!=prev:
                        collapsed.append(fc)
                        tc.append(tcid)
                        prev=fc
                tone_char_preds.extend(tc)
            tone_char_preds_tensor = torch.tensor(tone_char_preds, device=base_logits.device)
            preds_str = converter.decode_with_tones(base_idx.data, preds_size.data, tone_char_preds_tensor)
        else:
            preds_str = base_strs
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