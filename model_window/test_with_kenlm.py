import torch
import torch.nn.functional as F
import os
import re
import json
import numpy as np
import kenlm
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict
import editdistance

class KenLMTextScorer:
    """
    KenLM-based text scorer for CTC beam search outputs
    """
    def __init__(self, kenlm_model_path):
        print(f"Loading KenLM model: {kenlm_model_path}")
        self.model = kenlm.Model(kenlm_model_path)
    def score(self, text):
        return self.model.score(text, bos=True, eos=True)

def simple_ctc_beam_search_with_lm(log_probs, converter, lm_scorer, beam_size=5):
    """
    Simple CTC beam search decoder with KenLM scoring for a single sample (log_probs: [T, C])
    Returns best decoded string
    """
    T, C = log_probs.shape
    beams = [([], 0.0)]  # (sequence, log_prob)
    for t in range(T):
        new_beams = []
        probs = log_probs[t].cpu().numpy()
        top_c = np.argsort(probs)[-beam_size:][::-1]
        for seq, score in beams:
            for c in top_c:
                new_seq = seq + [c]
                new_score = score + probs[c]
                new_beams.append((new_seq, new_score))
        # Keep top beam_size beams
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        beams = new_beams
    # Convert to text, remove repeats and blanks (assume blank=0)
    candidates = []
    for seq, score in beams:
        text = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:
                text.append(idx)
            prev = idx
        decoded = converter.decode(np.array(text), np.array([len(text)]))
        if decoded and decoded[0]:
            candidates.append((decoded[0], score))
    # Score with KenLM
    lm_scores = [lm_scorer.score(cand[0]) for cand in candidates]
    best_idx = int(np.argmax(lm_scores)) if lm_scores else 0
    return candidates[best_idx][0] if candidates else ""

def validation_with_kenlm(model, criterion, evaluation_loader, converter, lm_scorer):
    model.eval()
    norm_ED = 0
    norm_ED_wer = 0
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
        with torch.no_grad():
            preds = model(image)
            preds = preds.float()
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds_log = preds.permute(1, 0, 2).log_softmax(2)
            torch.backends.cudnn.enabled = False
            cost = criterion(preds_log, text_for_loss,
                             preds_size, length_for_loss).mean()
            torch.backends.cudnn.enabled = True
            preds_str = []
            for b in range(batch_size):
                log_probs = preds_log[:, b, :]
                best_str = simple_ctc_beam_search_with_lm(log_probs, converter, lm_scorer, beam_size=5)
                preds_str.append(best_str)
        valid_loss += cost.item()
        count += 1
        all_preds_str.extend(preds_str)
        all_labels.extend(labels)
        # Calculate metrics for predictions
        for pred_cer, gt_cer in zip(preds_str, labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                norm_ED += tmp_ED / float(len(gt_cer))
            length_of_gt += len(gt_cer)
        # Calculate WER
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
            length_of_gt_wer += len(gt_wer)
        if i % 10 == 0:
            print(f"Batch {i+1}/{len(evaluation_loader)} processed")
    val_loss = valid_loss / count
    CER = norm_ED / float(count)
    WER = norm_ED_wer / float(count)
    return val_loss, CER, WER, all_preds_str, all_labels

def main():
    args = option.get_args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    # Load HTR model
    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    pth_path = args.resume_checkpoint
    logger.info('Loading HTR checkpoint from {}'.format(pth_path))
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
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    # Load KenLM model
    kenlm_model_path = getattr(args, 'kenlm_model_path', 'your_model.arpa')
    lm_scorer = KenLMTextScorer(kenlm_model_path)
    # Run validation with KenLM scoring
    model.eval()
    with torch.no_grad():
        val_loss, cer, wer, preds, labels = validation_with_kenlm(
            model, criterion, test_loader, converter, lm_scorer)
    logger.info('=' * 80)
    logger.info('KENLM-BEAMSEARCH RESULTS:')
    logger.info(f'Test loss: {val_loss:0.3f} \t CER: {cer:0.4f} \t WER: {wer:0.4f}')
    logger.info('=' * 80)
    # Save detailed results
    results_file = os.path.join(args.save_dir, 'kenlm_correction_results.json')
    detailed_results = {
        'cer': float(cer),
        'wer': float(wer),
        'predictions': {
            'kenlm': preds,
            'ground_truth': labels
        }
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    logger.info(f'Detailed results saved to: {results_file}')
    # Save sample comparisons
    sample_file = os.path.join(args.save_dir, 'sample_kenlm_corrections.txt')
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("Sample KenLM Corrections:\n")
        f.write("=" * 80 + "\n")
        for i in range(min(20, len(labels))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Ground Truth: {labels[i]}\n")
            f.write(f"KenLM BeamSearch: {preds[i]}\n")
            orig_cer = editdistance.eval(preds[i], labels[i]) / max(len(labels[i]), 1)
            f.write(f"CER: {orig_cer:.4f}\n")
            f.write("-" * 40 + "\n")
    logger.info(f'Sample KenLM corrections saved to: {sample_file}')

if __name__ == '__main__':
    main()
