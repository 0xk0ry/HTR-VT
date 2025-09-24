import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import math


def _ctc_beam_search_single(log_probs, beam_size, blank=0):
    """CTC prefix beam search for a single sequence.

    Args:
        log_probs: Tensor [T, V] of log-softmax probabilities.
        beam_size: int, number of beams to keep.
        blank: int, index of the CTC blank token.

    Returns:
        List of (prefix as tuple[int], log_prob) sorted by log_prob desc.
    """
    T, V = log_probs.size()
    # Each prefix is tracked with (p_b, p_nb) log-probabilities
    beams = {(): (0.0, float('-inf'))}  # empty prefix: p_b=0, p_nb=-inf in log-space

    for t in range(T):
        lp_t = log_probs[t]  # [V]
        next_beams = {}
        # Pre-select top-K symbols at this timestep to prune branching
        topk = min(beam_size, V)
        topk_logp, topk_idx = torch.topk(lp_t, topk)
        topk = list(zip(topk_idx.tolist(), topk_logp.tolist()))

        for prefix, (p_b, p_nb) in beams.items():
            # Extend with blank
            lp_blank = lp_t[blank].item()
            nb = next_beams.get(prefix, (float('-inf'), float('-inf')))
            nb_p_b = logsumexp2(nb[0], p_b + lp_blank)
            nb_p_b = logsumexp2(nb_p_b, p_nb + lp_blank)
            next_beams[prefix] = (nb_p_b, nb[1])

            # Extend with non-blank candidates
            for s, lp_s in topk:
                if s == blank:
                    continue
                new_prefix = prefix + (s,)
                # If last label is same as s, only p_b contributes; else both
                if len(prefix) > 0 and s == prefix[-1]:
                    pnb = p_b + lp_s
                else:
                    pnb = logsumexp2(p_b + lp_s, p_nb + lp_s)
                nb2 = next_beams.get(new_prefix, (float('-inf'), float('-inf')))
                next_beams[new_prefix] = (nb2[0], logsumexp2(nb2[1], pnb))

        # Prune to beam_size by total log-prob logsumexp(p_b, p_nb)
        def total_logp(v):
            return logsumexp2(v[0], v[1])
        beams = dict(sorted(next_beams.items(), key=lambda kv: total_logp(kv[1]), reverse=True)[:beam_size])

    # Finalize beams
    finalized = []
    for prefix, (p_b, p_nb) in beams.items():
        finalized.append((prefix, logsumexp2(p_b, p_nb)))
    finalized.sort(key=lambda x: x[1], reverse=True)
    return finalized


def logsumexp2(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)) for scalars."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def ctc_beam_search_batch(log_probs, beam_size, converter, blank=0):
    """Decode a batch using CTC beam search.

    Args:
        log_probs: [T, B, V] tensor of log-softmax outputs.
        beam_size: beam width.
        converter: CTCLabelConverter for mapping indices to chars.
        blank: blank index in vocabulary (default 0).

    Returns:
        List[str] decoded sequences for each item in batch.
    """
    T, B, V = log_probs.size()
    results = []
    for b in range(B):
        seq = log_probs[:, b, :].detach().cpu()
        beams = _ctc_beam_search_single(seq, beam_size=beam_size, blank=blank)
        best_prefix = beams[0][0] if beams else ()
        # Collapse repeats as CTC finalization already accounts for repeats via prefix prob; still safe
        chars = []
        prev = None
        for idx in best_prefix:
            if idx != blank and idx < len(converter.character) and idx != prev:
                chars.append(converter.character[idx])
            prev = idx
        results.append(''.join(chars))
    return results
import editdistance


def validation(model, criterion, evaluation_loader, converter, decode_method: str = 'greedy', beam_size: int = 5):
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

        preds = model(image)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2).log_softmax(2)

        torch.backends.cudnn.enabled = False
        cost = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()
        torch.backends.cudnn.enabled = True

        if decode_method == 'beam_search' and beam_size > 1:
            preds_str = ctc_beam_search_batch(preds, beam_size=beam_size, converter=converter, blank=0)
        else:
            # Greedy (default)
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

    return val_loss, CER, WER, all_preds_str, all_labels