#!/usr/bin/env python3
"""
Quick inference script with RoBERTa-based text correction
Usage: python quick_inference_llm.py --image_path path/to/image.jpg
"""

import torch
import torch.nn.functional as F
import os
import re
import argparse
import cv2
import numpy as np
from PIL import Image
from transformers import RobertaTokenizer, RobertaForMaskedLM
from utils import utils
from data import transform
from model import HTR_VT
from collections import OrderedDict


class RobertaTextCorrector:
    """Lightweight RoBERTa-based text correction for inference"""

    def __init__(self, model_name="roberta-large", device="cuda", confidence_threshold=0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold

        print(f"Loading RoBERTa model: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def get_word_candidates(self, text, position, top_k=3):
        """Get word candidates for a specific position using masked language modeling"""
        words = text.split()
        if position >= len(words) or not words:
            return [(text, 0.0)]

        # Create masked version
        masked_words = words.copy()
        original_word = masked_words[position]
        masked_words[position] = self.mask_token
        masked_text = " ".join(masked_words)

        try:
            # Tokenize and find mask position
            inputs = self.tokenizer(
                masked_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            # Find mask token position
            mask_token_indices = (
                inputs['input_ids'] == self.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_token_indices) == 0:
                return [(original_word, 0.0)]

            mask_token_logits = predictions[0, mask_token_indices[0], :]
            probs = F.softmax(mask_token_logits, dim=-1)

            # Get top-k predictions
            top_k_tokens = torch.topk(probs, min(top_k, probs.size(0)))
            candidates = []

            for i in range(top_k_tokens.indices.size(0)):
                token_id = top_k_tokens.indices[i].item()
                probability = top_k_tokens.values[i].item()
                token = self.tokenizer.decode([token_id]).strip()

                # Filter out special tokens and subwords
                if (not token.startswith('<') and
                    not token.startswith('Ġ') and
                    token.replace('Ġ', '').isalpha() and
                        len(token.strip()) > 0):
                    clean_token = token.replace('Ġ', '').strip()
                    candidates.append((clean_token, probability))

            # Include original word if not in candidates and it's valid
            if original_word and len(original_word.strip()) > 0:
                original_found = any(
                    cand[0].lower() == original_word.lower() for cand in candidates)
                if not original_found:
                    candidates.append((original_word, 0.0))

            return candidates[:top_k] if candidates else [(original_word, 0.0)]

        except Exception as e:
            print(f"Error in get_word_candidates: {e}")
            return [(original_word, 0.0)]

    def correct_text(self, text, max_iterations=2):
        """Correct text using RoBERTa with limited iterations for speed"""
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
                candidates = self.get_word_candidates(
                    current_text, pos, top_k=2)

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
                        break  # Take first improvement for speed

            if not improved:
                break

            current_text = best_text
            words = current_text.split()

        return best_text

    def score_text(self, text):
        """Score text using RoBERTa perplexity"""
        if not text or len(text.strip()) == 0:
            return float('-inf')

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
                return -loss
        except:
            return float('-inf')


def load_htr_model(checkpoint_path, nb_cls=90, img_size=(512, 64), device="cuda"):
    """Load the HTR model from checkpoint"""
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size[::-1])

    print(f'Loading HTR checkpoint from {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=(512, 64)):
    """Preprocess image for HTR model"""
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    else:
        image = image_path

    # Resize image
    h, w = image.shape
    target_w, target_h = img_size

    # Maintain aspect ratio
    aspect_ratio = w / h
    if aspect_ratio > target_w / target_h:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    # Normalize
    normalized = (padded.astype(np.float32) - 127.5) / 127.5

    # Convert to tensor
    tensor = torch.from_numpy(normalized).unsqueeze(
        0).unsqueeze(0)  # [1, 1, H, W]

    return tensor


def create_character_dict():
    """Create a simple character dictionary for CTC decoding"""
    # Standard character set for handwriting recognition
    chars = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    char_dict = {i+1: char for i, char in enumerate(chars)}  # 0 is blank
    char_dict[0] = '[blank]'
    return char_dict


def ctc_decode(logits, char_dict):
    """Simple CTC decoding"""
    # Get the most likely characters
    _, preds_index = logits.max(2)  # [T, B]
    preds_index = preds_index.squeeze(1)  # [T]

    # Remove blanks and repetitions
    decoded_chars = []
    prev_char = 0

    for i in range(len(preds_index)):
        char_idx = preds_index[i].item()
        if char_idx != 0 and char_idx != prev_char:  # Not blank and not repetition
            if char_idx in char_dict:
                decoded_chars.append(char_dict[char_idx])
        prev_char = char_idx

    return ''.join(decoded_chars)


def infer_single_image(image_path, htr_model, text_corrector, char_dict, device="cuda", img_size=(512, 64)):
    """Perform inference on a single image"""
    # Preprocess image
    image_tensor = preprocess_image(image_path, img_size).to(device)

    # HTR inference
    with torch.no_grad():
        logits = htr_model(image_tensor)  # [B, T, num_classes]
        logits = logits.permute(1, 0, 2)  # [T, B, num_classes]

    # CTC decode
    original_text = ctc_decode(logits, char_dict)

    # LLM correction
    corrected_text = text_corrector.correct_text(original_text)

    return original_text, corrected_text


def main():
    parser = argparse.ArgumentParser(
        description='HTR Inference with RoBERTa Correction')
    parser.add_argument('--image_path', type=str,
                        required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str,
                        default='./best_CER.pth', help='Path to HTR checkpoint')
    parser.add_argument('--img_size', type=int, nargs=2,
                        default=[512, 64], help='Image size [width, height]')
    parser.add_argument('--nb_cls', type=int, default=90,
                        help='Number of character classes')
    parser.add_argument('--roberta_model', type=str,
                        default='roberta-large', help='RoBERTa model name')
    parser.add_argument('--confidence_threshold', type=float,
                        default=0.5, help='RoBERTa confidence threshold')
    parser.add_argument('--device', type=str,
                        default='cuda', help='Device to run on')
    parser.add_argument('--disable_correction',
                        action='store_true', help='Disable LLM correction')

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} not found")
        return

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load HTR model
    print("Loading HTR model...")
    htr_model = load_htr_model(
        args.checkpoint, args.nb_cls, tuple(args.img_size), device)

    # Create character dictionary
    char_dict = create_character_dict()

    # Load text corrector if not disabled
    text_corrector = None
    if not args.disable_correction:
        print("Loading RoBERTa text corrector...")
        text_corrector = RobertaTextCorrector(
            model_name=args.roberta_model,
            device=device,
            confidence_threshold=args.confidence_threshold
        )
    else:
        print("LLM correction disabled")

    # Perform inference
    print(f"Processing image: {args.image_path}")
    try:
        if text_corrector is not None:
            original_text, corrected_text = infer_single_image(
                args.image_path, htr_model, text_corrector, char_dict, device, tuple(
                    args.img_size)
            )

            print("\n" + "="*60)
            print("RESULTS:")
            print("="*60)
            print(f"Original CTC:      '{original_text}'")
            print(f"RoBERTa Corrected: '{corrected_text}'")
            print("="*60)
        else:
            # Only CTC decoding
            image_tensor = preprocess_image(
                args.image_path, tuple(args.img_size)).to(device)
            with torch.no_grad():
                logits = htr_model(image_tensor).permute(1, 0, 2)
            original_text = ctc_decode(logits, char_dict)

            print("\n" + "="*60)
            print("RESULTS:")
            print("="*60)
            print(f"CTC Output: '{original_text}'")
            print("="*60)

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
