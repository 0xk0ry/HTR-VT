import torch
import torch.nn.functional as F
import os
import re
import json
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM
import valid
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict
import editdistance


class RobertaTextCorrector:
    """
    RoBERTa-based text correction module for post-processing CTC outputs
    """
    def __init__(self, model_name="roberta-large", device="cuda", confidence_threshold=0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading RoBERTa model: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Special tokens
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        
    def get_word_candidates(self, text, position, top_k=5):
        """
        Get word candidates for a specific position using masked language modeling
        """
        words = text.split()
        if position >= len(words):
            return [(words[position] if position < len(words) else "", 0.0)]
            
        # Create masked version
        masked_words = words.copy()
        original_word = masked_words[position]
        masked_words[position] = self.mask_token
        masked_text = " ".join(masked_words)
        
        # Tokenize and find mask position
        inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            
        # Find mask token position
        mask_token_indices = (inputs['input_ids'] == self.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_token_indices) == 0:
            return [(original_word, 0.0)]
            
        mask_token_logits = predictions[0, mask_token_indices[0], :]
        probs = F.softmax(mask_token_logits, dim=-1)
        
        # Get top-k predictions
        top_k_tokens = torch.topk(probs, top_k)
        candidates = []
        
        for i in range(top_k):
            token_id = top_k_tokens.indices[i].item()
            probability = top_k_tokens.values[i].item()
            token = self.tokenizer.decode([token_id]).strip()
            
            # Filter out special tokens and subwords
            if not token.startswith('<') and not token.startswith('Ġ') and token.isalpha():
                candidates.append((token, probability))
                
        # Include original word if not in candidates
        original_found = any(cand[0].lower() == original_word.lower() for cand in candidates)
        if not original_found:
            candidates.append((original_word, 0.0))
            
        return candidates[:top_k]
    
    def correct_text_iterative(self, text, max_iterations=3):
        """
        Iteratively correct text using RoBERTa
        """
        if not text or len(text.strip()) == 0:
            return text
            
        current_text = text.strip()
        words = current_text.split()
        
        if len(words) == 0:
            return text
            
        best_text = current_text
        best_score = self.score_text(current_text)
        
        for iteration in range(max_iterations):
            improved = False
            
            for pos in range(len(words)):
                candidates = self.get_word_candidates(current_text, pos, top_k=3)
                
                for candidate_word, confidence in candidates:
                    if confidence < self.confidence_threshold:
                        continue
                        
                    # Create candidate text
                    test_words = words.copy()
                    test_words[pos] = candidate_word
                    candidate_text = " ".join(test_words)
                    
                    # Score the candidate
                    candidate_score = self.score_text(candidate_text)
                    
                    if candidate_score > best_score:
                        best_text = candidate_text
                        best_score = candidate_score
                        improved = True
                        
            if not improved:
                break
                
            current_text = best_text
            words = current_text.split()
                
        return best_text
    
    def score_text(self, text):
        """
        Score text using RoBERTa perplexity
        """
        if not text or len(text.strip()) == 0:
            return float('-inf')
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
                # Convert loss to score (lower loss = higher score)
                return -loss
        except:
            return float('-inf')
    
    def correct_batch(self, texts):
        """
        Correct a batch of texts
        """
        corrected_texts = []
        for text in texts:
            corrected = self.correct_text_iterative(text)
            corrected_texts.append(corrected)
        return corrected_texts


def validation_with_llm_correction(model, criterion, evaluation_loader, converter, text_corrector):
    """
    Validation function with LLM-based text correction
    """
    model.eval()
    
    norm_ED = 0
    norm_ED_wer = 0
    norm_ED_corrected = 0
    norm_ED_wer_corrected = 0
    
    tot_ED = 0
    tot_ED_wer = 0
    tot_ED_corrected = 0
    tot_ED_wer_corrected = 0
    
    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    count = 0
    
    all_preds_str = []
    all_corrected_preds = []
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
            cost = criterion(preds_log, text_for_loss, preds_size, length_for_loss).mean()
            torch.backends.cudnn.enabled = True
            
            _, preds_index = preds_log.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
            
        valid_loss += cost.item()
        count += 1
        
        # Apply LLM correction
        corrected_preds = text_corrector.correct_batch(preds_str)
        
        all_preds_str.extend(preds_str)
        all_corrected_preds.extend(corrected_preds)
        all_labels.extend(labels)
        
        # Calculate metrics for original predictions
        for pred_cer, gt_cer in zip(preds_str, labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                norm_ED += tmp_ED / float(len(gt_cer))
            tot_ED += tmp_ED
            length_of_gt += len(gt_cer)
            
        # Calculate metrics for corrected predictions
        for pred_cer, gt_cer in zip(corrected_preds, labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED_corrected += 1
            else:
                norm_ED_corrected += tmp_ED / float(len(gt_cer))
            tot_ED_corrected += tmp_ED
            
        # Calculate WER for original predictions
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
            
        # Calculate WER for corrected predictions
        for pred_wer, gt_wer in zip(corrected_preds, labels):
            pred_wer = utils.format_string_for_wer(pred_wer)
            gt_wer = utils.format_string_for_wer(gt_wer)
            pred_wer = pred_wer.split(" ")
            gt_wer = gt_wer.split(" ")
            tmp_ED_wer = editdistance.eval(pred_wer, gt_wer)
            
            if len(gt_wer) == 0:
                norm_ED_wer_corrected += 1
            else:
                norm_ED_wer_corrected += tmp_ED_wer / float(len(gt_wer))
            tot_ED_wer_corrected += tmp_ED_wer
            
        if i % 10 == 0:
            print(f"Batch {i+1}/{len(evaluation_loader)} processed")
            
    val_loss = valid_loss / count
    CER_original = tot_ED / float(length_of_gt)
    WER_original = tot_ED_wer / float(length_of_gt_wer)
    CER_corrected = tot_ED_corrected / float(length_of_gt)
    WER_corrected = tot_ED_wer_corrected / float(length_of_gt_wer)
    
    return (val_loss, CER_original, WER_original, CER_corrected, WER_corrected, 
            all_preds_str, all_corrected_preds, all_labels)


def main():
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

    # Initialize RoBERTa text corrector
    logger.info('Loading RoBERTa Large model for text correction...')
    text_corrector = RobertaTextCorrector(
        model_name="roberta-large", 
        device=device,
        confidence_threshold=getattr(args, 'roberta_confidence_threshold', 0.5)
    )

    # Run validation with LLM correction
    model.eval()
    with torch.no_grad():
        results = validation_with_llm_correction(
            model, criterion, test_loader, converter, text_corrector
        )
        
        (val_loss, cer_original, wer_original, cer_corrected, wer_corrected, 
         preds_original, preds_corrected, labels) = results

    # Log results
    logger.info('=' * 80)
    logger.info('ORIGINAL CTC RESULTS:')
    logger.info(f'Test loss: {val_loss:0.3f} \t CER: {cer_original:0.4f} \t WER: {wer_original:0.4f}')
    logger.info('=' * 80)
    logger.info('ROBERTA-CORRECTED RESULTS:')
    logger.info(f'Test loss: {val_loss:0.3f} \t CER: {cer_corrected:0.4f} \t WER: {wer_corrected:0.4f}')
    logger.info('=' * 80)
    logger.info('IMPROVEMENT:')
    cer_improvement = ((cer_original - cer_corrected) / cer_original) * 100 if cer_original > 0 else 0
    wer_improvement = ((wer_original - wer_corrected) / wer_original) * 100 if wer_original > 0 else 0
    logger.info(f'CER improvement: {cer_improvement:0.2f}%')
    logger.info(f'WER improvement: {wer_improvement:0.2f}%')
    
    # Save detailed results
    results_file = os.path.join(args.save_dir, 'llm_correction_results.json')
    detailed_results = {
        'original_cer': float(cer_original),
        'original_wer': float(wer_original),
        'corrected_cer': float(cer_corrected),
        'corrected_wer': float(wer_corrected),
        'cer_improvement_percent': float(cer_improvement),
        'wer_improvement_percent': float(wer_improvement),
        'predictions': {
            'original': preds_original,
            'corrected': preds_corrected,
            'ground_truth': labels
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    logger.info(f'Detailed results saved to: {results_file}')
    
    # Save sample comparisons
    sample_file = os.path.join(args.save_dir, 'sample_corrections.txt')
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("Sample Text Corrections:\n")
        f.write("=" * 80 + "\n")
        
        for i in range(min(20, len(labels))):  # Show first 20 samples
            f.write(f"Sample {i+1}:\n")
            f.write(f"Ground Truth: {labels[i]}\n")
            f.write(f"Original CTC: {preds_original[i]}\n")
            f.write(f"RoBERTa Corrected: {preds_corrected[i]}\n")
            
            # Calculate individual improvements
            orig_cer = editdistance.eval(preds_original[i], labels[i]) / max(len(labels[i]), 1)
            corr_cer = editdistance.eval(preds_corrected[i], labels[i]) / max(len(labels[i]), 1)
            f.write(f"Original CER: {orig_cer:.4f}, Corrected CER: {corr_cer:.4f}\n")
            f.write("-" * 40 + "\n")
            
    logger.info(f'Sample corrections saved to: {sample_file}')


if __name__ == '__main__':
    args = option.get_args_parser()
    
    # Add RoBERTa-specific arguments
    if not hasattr(args, 'roberta_confidence_threshold'):
        args.roberta_confidence_threshold = 0.5
        
    main()
