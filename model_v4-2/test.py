import torch
import torch.nn.functional as F

import os
import re
import json
import editdistance
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT  # type: ignore
from collections import OrderedDict


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    pth_path = args.resume
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()

    logger.info('Loading test loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)

    test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    # Tone-aware evaluation (compose tones on decoded predictions)
    model.eval()
    vowel_idxs = utils.vowel_indices_from_converter(converter)
    tau_v = getattr(args, 'tau_v', 0.5)
    kappa = getattr(args, 'tone_kappa', 0.2)
    use_tone = getattr(args, 'use_tone_head', True)

    norm_ED = 0.0
    norm_ED_wer = 0.0
    tot_ED = 0
    tot_ED_wer = 0
    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    count = 0

    with torch.no_grad():
        for image_tensors, labels in test_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.cuda()

            # Encode labels for CTC loss
            text_for_loss, length_for_loss = converter.encode(labels)

            outputs = model(image)
            # Support both (base, tone) and legacy single output
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                base_logits, tone_logits = outputs
            else:
                base_logits, tone_logits = outputs, None

            # Compute CTC loss on base head
            preds_size = torch.IntTensor([base_logits.size(1)] * batch_size)
            log_probs = base_logits.permute(1, 0, 2).log_softmax(2)
            torch.backends.cudnn.enabled = False
            cost = criterion(log_probs, text_for_loss, preds_size, length_for_loss).mean()
            torch.backends.cudnn.enabled = True
            valid_loss += float(cost.item())
            count += 1

            # Greedy decode base head
            _, preds_index = log_probs.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str_base = converter.decode(preds_index.data, preds_size.data)

            # Compose tones per sample
            base_probs = F.softmax(base_logits, dim=-1)
            if tone_logits is not None and use_tone:
                tone_probs = F.softmax(tone_logits, dim=-1)
            else:
                tone_probs = None

            for b in range(batch_size):
                pred_base = preds_str_base[b]
                label = labels[b]

                # Build framewise argmax path (T,)
                frame_labels = base_probs[b].argmax(dim=-1)  # (T,)
                T = frame_labels.numel()
                # Gate by vowel probability
                gate = (base_probs[b, :, vowel_idxs].sum(dim=-1) >= tau_v).float() if len(vowel_idxs) > 0 else torch.zeros(T, device=base_probs.device)

                # Map nonblank repeated labels to spans and compose per char
                composed_chars = []
                current_k = None
                start_t = 0
                for t in range(T + 1):
                    k = int(frame_labels[t].item()) if t < T else -1  # sentinel to flush
                    if current_k is None:
                        if t < T and k != 0:
                            current_k = k
                            start_t = t
                        continue
                    if t == T or k == 0 or k != current_k:
                        # Close span for current_k over [start_t, t)
                        if current_k != 0 and current_k < len(converter.character):
                            ch = converter.character[current_k]
                            if tone_probs is not None and utils.is_vietnamese_vowel(ch):
                                tone_id = utils.aggregate_tone_over_span(tone_logits[b], gate, start_t, t, margin_kappa=kappa)
                                ch = utils.apply_tone_to_char(ch, int(tone_id))
                            composed_chars.append(ch)
                        # Start new if next label is nonblank
                        if t < T and k != 0:
                            current_k = k
                            start_t = t
                        else:
                            current_k = None
                    # else: continue accumulating same label

                pred_composed = ''.join(composed_chars)

                # Compute CER/WER against ground truth
                tmp_ED = editdistance.eval(pred_composed, label)
                if len(label) == 0:
                    norm_ED += 1
                else:
                    norm_ED += tmp_ED / float(len(label))
                tot_ED += tmp_ED
                length_of_gt += len(label)

                pred_w = utils.format_string_for_wer(pred_composed)
                gt_w = utils.format_string_for_wer(label)
                pred_w = pred_w.split(' ')
                gt_w = gt_w.split(' ')
                tmp_ED_wer = editdistance.eval(pred_w, gt_w)
                if len(gt_w) == 0:
                    norm_ED_wer += 1
                else:
                    norm_ED_wer += tmp_ED_wer / float(len(gt_w))
                tot_ED_wer += tmp_ED_wer
                length_of_gt_wer += len(gt_w)

    val_loss = valid_loss / max(1, count)
    CER = tot_ED / float(max(1, length_of_gt))
    WER = tot_ED_wer / float(max(1, length_of_gt_wer))

    logger.info(f'Test. loss : {val_loss:0.3f} \t CER : {CER:0.4f} \t WER : {WER:0.4f}')


if __name__ == '__main__':
    args = option.get_args_parser()
    main()

