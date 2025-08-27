import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import argparse

from utils import utils
import editdistance
from utils import vn_tags


def _compose_if_dual(preds_dict, converter, args):
    base_logits = preds_dict['base']
    mod_logits = preds_dict['mod']
    tone_logits = preds_dict['tone']
    B, T, _ = base_logits.shape
    # Decode base via greedy CTC collapse
    base_logp = base_logits.permute(1,0,2).log_softmax(2)
    _, base_idx = base_logp.max(2)
    base_idx = base_idx.transpose(1,0).contiguous().view(-1)
    preds_size = torch.IntTensor([base_logits.size(1)]*B)
    base_strs = converter.decode(base_idx.data, preds_size.data)
    # Greedy frame alignment for tags: collapse consecutive frames matching base sequence order
    base_argmax = base_logits.argmax(dim=2)  # [B,T]
    mod_ids = mod_logits.argmax(dim=2)       # [B,T]
    tone_ids = tone_logits.argmax(dim=2)     # [B,T]
    composed=[]
    for b in range(B):
        frames = base_argmax[b]
        non_blank = frames != 0
        prev = torch.cat([torch.tensor([0],device=frames.device), frames[:-1]])
        change = (frames != prev) & non_blank
        seg_starts = torch.nonzero(change, as_tuple=True)[0]
        seg_vals = frames[seg_starts]
        mod_seq=[]; tone_seq=[]; target_order = []
        for si in range(len(seg_starts)):
            st=seg_starts[si]
            en=seg_starts[si+1] if si+1 < len(seg_starts) else len(frames)
            seg_mod = mod_ids[b, st:en].mode()[0].item()
            seg_tone = tone_ids[b, st:en].mode()[0].item()
            target_order.append(seg_vals[si].item())
            mod_seq.append(seg_mod)
            tone_seq.append(seg_tone)
        # Map target_order (CTC classes) back to characters via converter.character
        # Remove blanks (0) and collapse duplicates already handled.
        base_decoded = base_strs[b]
        # If vowel-only-tags, zero-out consonant tags
        if args.vowel_only_tags:
            base_chars = list(base_decoded)
            for i,ch in enumerate(base_chars):
                if not vn_tags.is_vowel(ch):
                    if i < len(mod_seq): mod_seq[i] = 0
                    if i < len(tone_seq): tone_seq[i] = 0
        composed.append(vn_tags.compose_str(base_decoded, mod_seq, tone_seq))
    return base_strs, composed


def validation(model, criterion, evaluation_loader, converter, args=None):
    """Validation/evaluation.
    Returns:
        val_loss, CER_full, WER_full, preds_full, labels, CER_base, WER_base
        (For single-head models, CER_base/WER_base == CER_full/WER_full)
    """

    norm_ED = 0
    norm_ED_wer = 0
    tot_ED = 0
    tot_ED_wer = 0

    # Base-only accumulators (dual-head)
    norm_ED_base = 0
    norm_ED_wer_base = 0
    tot_ED_base = 0
    tot_ED_wer_base = 0
    length_of_gt_base = 0
    length_of_gt_wer_base = 0

    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    count = 0
    all_preds_str = []
    all_labels = []
    dual = getattr(args, 'use_dual_head', False)

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.cuda()

        enc = converter.encode(labels)
        # Support DualLabelConverter (5-tuple) and CTCLabelConverter (2-tuple)
        if isinstance(enc, (list, tuple)) and len(enc) == 5:
            text_for_loss, length_for_loss = enc[0], enc[1]
        else:
            text_for_loss, length_for_loss = enc

        preds_out = model(image)
        if isinstance(preds_out, dict):
            base_logits = preds_out['base']
            preds_size = torch.IntTensor([base_logits.size(1)] * batch_size)
            base_logp = base_logits.permute(1,0,2).log_softmax(2)
            torch.backends.cudnn.enabled = False
            cost = criterion(base_logp, text_for_loss, preds_size, length_for_loss).mean()
            torch.backends.cudnn.enabled = True
            _, preds_index = base_logp.max(2)
            preds_index = preds_index.transpose(1,0).contiguous().view(-1)
            base_strs = converter.decode(preds_index.data, preds_size.data)
            base_only, composed = _compose_if_dual(preds_out, converter, args)
            eval_strings = composed  # full (with diacritics)
        else:
            preds = preds_out
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2).log_softmax(2)
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()
            torch.backends.cudnn.enabled = True
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
            eval_strings = preds_str
            base_strs = preds_str  # single-head: base == full
            base_only = base_strs
            composed = eval_strings
        valid_loss += cost.item()
        count += 1

        all_preds_str.extend(eval_strings)
        all_labels.extend(labels)

        for pred_cer, gt_cer in zip(eval_strings, labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                norm_ED += tmp_ED / float(len(gt_cer))
            tot_ED += tmp_ED
            length_of_gt += len(gt_cer)

        for pred_wer, gt_wer in zip(eval_strings, labels):
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

        # Base-only metrics (only meaningful if dual-head and labels contain diacritics)
        if dual:
            base_gts = [vn_tags.decompose_str(lbl)[0] for lbl in labels]
            for pred_base, gt_base in zip(base_only, base_gts):
                tmp_ED_b = editdistance.eval(pred_base, gt_base)
                if len(gt_base) == 0:
                    norm_ED_base += 1
                else:
                    norm_ED_base += tmp_ED_b / float(len(gt_base))
                tot_ED_base += tmp_ED_b
                length_of_gt_base += len(gt_base)
            # WER base (word-level over base strings)
            for pred_bwer, gt_bwer in zip(base_only, base_gts):
                pred_bwer_fmt = utils.format_string_for_wer(pred_bwer).split(" ")
                gt_bwer_fmt = utils.format_string_for_wer(gt_bwer).split(" ")
                tmp_ED_bwer = editdistance.eval(pred_bwer_fmt, gt_bwer_fmt)
                if len(gt_bwer_fmt) == 0:
                    norm_ED_wer_base += 1
                else:
                    norm_ED_wer_base += tmp_ED_bwer / float(len(gt_bwer_fmt))
                tot_ED_wer_base += tmp_ED_bwer
                length_of_gt_wer_base += len(gt_bwer_fmt)

    val_loss = valid_loss / count
    CER = tot_ED / float(length_of_gt)
    WER = tot_ED_wer / float(length_of_gt_wer)
    if dual and length_of_gt_base > 0 and length_of_gt_wer_base > 0:
        CER_base = tot_ED_base / float(length_of_gt_base)
        WER_base = tot_ED_wer_base / float(length_of_gt_wer_base)
    else:
        CER_base, WER_base = CER, WER
    return val_loss, CER, WER, all_preds_str, all_labels, CER_base, WER_base