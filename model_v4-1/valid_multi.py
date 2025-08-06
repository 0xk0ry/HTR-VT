import torch
import torch.backends.cudnn as cudnn
import editdistance
import unicodedata


def validation(
    model,
    criterion_ctc,
    criterion_diac,
    evaluation_loader,
    converter_base,      # converter for base‐alphabet CTC decode
    converter_diac,      # converter for diacritic CTC decode
    max_seq_len,
    alpha_ctc=0.7,
    window=2
):
    """ validation or evaluation for multi-task base+diacritic model """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    tot_ctc_loss = 0.0
    tot_diac_loss = 0.0
    count = 0

    tot_ED = 0
    tot_len = 0
    tot_ED_wer = 0
    tot_len_wer = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in evaluation_loader:
            B = images.size(0)
            images = images.to(device)

            # 1) Prepare base-char CTC targets
            text_base, len_base = converter_base.encode(labels)
            text_base = text_base.to(device)
            len_base = len_base.to(device)

            # 2) Prepare diacritic targets if you want to track diacritic loss
            #    (We can skip this if we only care about CTC loss in val)
            text_diac = converter_diac.encode(labels).to(device)

            # 3) Forward
            logp_base, logits_diac = model(images)
            # logp_base: (T, B, C_base)
            # logits_diac: (B, T, 6)

            # 4) Loss (optional)
            preds_size = torch.full((B,), logp_base.size(
                0), dtype=torch.int32, device=device)
            ctc_loss = criterion_ctc(
                logp_base, text_base, preds_size, len_base
            )
            # If tracking diacritic CE:
            B, T, K = logits_diac.size()
            diac_loss = criterion_diac(
                logits_diac.view(B*T, K),
                text_diac.view(B*T)
            )
            total_loss += alpha_ctc*ctc_loss + (1-alpha_ctc)*diac_loss

            tot_ctc_loss += ctc_loss.item()  # + (1-alpha_ctc)*diac_loss.item()
            tot_diac_loss += diac_loss.item()
            count += 1

            # 5) Greedy CTC decode + get emission times
            blank = 0  # your blank index
            base_ids = []
            time_idxs = []
            # compute per-frame argmax
            argmax_ids = logp_base.argmax(dim=-1)  # (T, B)
            T = argmax_ids.size(0)
            for b in range(B):
                prev = blank
                seq = []
                idxs = []
                for t in range(T):
                    c = argmax_ids[t, b].item()
                    if c != blank and c != prev:
                        seq.append(c)
                        idxs.append(t)
                    prev = c
                base_ids.append(seq)
                time_idxs.append(idxs)

            # 6) Diacritic window‐vote
            diac_probs = torch.softmax(logits_diac, dim=-1)  # (B, T, 6)
            final_pairs = []  # for recombination
            for b in range(B):
                pairs = []
                for i, cid in enumerate(base_ids[b]):
                    t = time_idxs[b][i]
                    start = max(0, t - window)
                    end = min(T, t + window + 1)
                    window_sum = diac_probs[b, start:end].sum(dim=0)  # (6,)
                    did = window_sum.argmax().item()
                    pairs.append((cid, did))
                final_pairs.append(pairs)

            # 7) Recombine into full Unicode strings
            preds_str = []
            for pairs in final_pairs:
                chars = []
                for cid, did in pairs:
                    base_ch = converter_base.idx2char[cid]
                    diac_ch = {
                        0: '', 1: '\u0300', 2: '\u0309',
                        3: '\u0303', 4: '\u0301', 5: '\u0323'
                    }[did]
                    chars.append(unicodedata.normalize(
                        'NFC', base_ch + diac_ch))
                preds_str.append(''.join(chars))

            # 8) Accumulate for CER/WER
            all_preds.extend(preds_str)
            all_labels.extend(labels)

            for pred, gt in zip(preds_str, labels):
                ed = editdistance.eval(pred, gt)
                tot_ED += ed
                tot_len += len(gt)
                # WER
                p_tokens = pred.split()
                g_tokens = gt.split()
                ed_w = editdistance.eval(p_tokens, g_tokens)
                tot_ED_wer += ed_w
                tot_len_wer += len(g_tokens)

    avg_ctc_loss = tot_ctc_loss / count
    avg_diac_loss = tot_diac_loss / count
    avg_avg_loss = total_loss / count
    CER = tot_ED / tot_len
    WER = tot_ED_wer / tot_len_wer

    return avg_avg_loss, avg_ctc_loss, avg_diac_loss, CER, WER, all_preds, all_labels
