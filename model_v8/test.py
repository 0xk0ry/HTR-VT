import torch

import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Use dynamic base charset length like training
    base_charset_str = utils.build_base_charset()
    nb_cls = len(base_charset_str) + 1
    model = HTR_VT.create_model(
        nb_cls=nb_cls, img_size=args.img_size[::-1])

    # pth_path = args.save_dir + '/best_CER.pth'
    pth_path = args.resume_checkpoint
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
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
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size)

    test_dataset = dataset.myLoadDS(
        args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    # Tone-aware converter (matches training pipeline)
    converter = utils.ToneLabelConverter(base_charset_str)
    criterion = torch.nn.CTCLoss(
        reduction='none', zero_infinity=True).to(device)

    model.eval()
    with torch.no_grad():
        (val_loss,
         cer_base,
         wer_base,
         cer_full,
         wer_full,
         ter,
         preds,
         labels) = valid.validation(model,
                                    criterion,
                                    test_loader,
                                    converter)

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER_base : {cer_base:0.4f} \t WER_base : {wer_base:0.4f} '
        f'\t CER : {cer_full:0.4f} \t WER : {wer_full:0.4f} \t TER : {ter:0.4f}')

    # Save predictions as JSON
    results = {
        "test_metrics": {
            "loss": float(val_loss),
            "cer_base": float(cer_base),
            "wer_base": float(wer_base),
            "cer": float(cer_full),
            "wer": float(wer_full),
            "ter": float(ter)
        },
        "predictions": []
    }

    # Helper functions for per-sample CER / WER
    def _levenshtein(a: str, b: str):
        # Early exits
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        # DP single row optimization
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                cur.append(min(prev[j] + 1,              # deletion
                               cur[j - 1] + 1,          # insertion
                               prev[j - 1] + cost))     # substitution
            prev = cur
        return prev[-1]

    def _cer(pred: str, gt: str):
        if len(gt) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return _levenshtein(pred, gt) / len(gt)

    def _wer(pred: str, gt: str):
        gt_words = gt.split()
        pred_words = pred.split()
        if len(gt_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        return _levenshtein(pred_words, gt_words) / len(gt_words)

    # Adapt Levenshtein to list of tokens (words)
    def _levenshtein(pred_tokens, gt_tokens):
        # Works for both list (words) and string (chars) due to iteration
        if pred_tokens == gt_tokens:
            return 0
        lp, lg = len(pred_tokens), len(gt_tokens)
        if lp == 0:
            return lg
        if lg == 0:
            return lp
        prev = list(range(lg + 1))
        for i in range(1, lp + 1):
            cur = [i]
            pi = pred_tokens[i - 1]
            for j in range(1, lg + 1):
                gj = gt_tokens[j - 1]
                cost = 0 if pi == gj else 1
                cur.append(
                    min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            prev = cur
        return prev[-1]

    # Re-map string-based levenshtein after redefining token version
    def _levenshtein_str(a: str, b: str):
        return _levenshtein(list(a), list(b))

    # Override char-based helpers to use token-aware underlying function
    def _cer(pred: str, gt: str):
        if len(gt) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return _levenshtein_str(pred, gt) / len(gt)

    def _wer(pred: str, gt: str):
        gt_words = gt.split()
        pred_words = pred.split()
        if len(gt_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        return _levenshtein(pred_words, gt_words) / len(gt_words)

    # Per-sample extended stats
    for i, (pred_full, label_full) in enumerate(zip(preds, labels)):
        if i < len(test_dataset.fns):
            img_path = test_dataset.fns[i]
            img_name = os.path.basename(img_path)
        else:
            img_path = None
            img_name = None
        # Decompose to base + tones to compute base metrics & tone mismatches
        pred_base, pred_tones = converter.vn_tags.decompose_str(pred_full)
        label_base, label_tones = converter.vn_tags.decompose_str(label_full)
        sample_cer_full = _cer(pred_full, label_full)
        sample_cer_base = _cer(pred_base, label_base)
        sample_wer_full = _wer(pred_full, label_full)
        sample_wer_base = _wer(pred_base, label_base)
        tone_len = len(label_tones)
        tone_err = 0
        for pos in range(tone_len):
            if pos >= len(pred_tones) or pred_tones[pos] != label_tones[pos]:
                tone_err += 1
        sample_ter = tone_err / tone_len if tone_len > 0 else 0.0
        results["predictions"].append({
            "sample_id": i + 1,
            "image_filename": img_name,
            "image_path": img_path,
            "prediction_full": pred_full,
            "prediction_base": pred_base,
            "ground_truth_full": label_full,
            "ground_truth_base": label_base,
            "match_full": pred_full == label_full,
            "match_base": pred_base == label_base,
            "cer_full": round(float(sample_cer_full), 6),
            "cer_base": round(float(sample_cer_base), 6),
            "wer_full": round(float(sample_wer_full), 6),
            "wer_base": round(float(sample_wer_base), 6),
            "ter": round(float(sample_ter), 6)
        })

    # Save to JSON file
    pred_file = os.path.join(args.save_dir, 'predictions.json')
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    args = option.get_args_parser()
    main()
