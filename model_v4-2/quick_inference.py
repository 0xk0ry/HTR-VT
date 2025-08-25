import torch
import torch.utils.data
from PIL import Image
import argparse

import os
import re
from utils import utils
from model import HTR_VT
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np


def load_model(checkpoint_path, nb_cls, img_size):
    """Load the trained model from checkpoint. If available, override nb_cls/img_size from ckpt args."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Loading model from: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Try to override shapes from checkpoint args for compatibility
    try:
        ck_args = ckpt.get('args', {})
        if isinstance(ck_args, dict):
            if 'nb_cls' in ck_args:
                nb_cls = int(ck_args['nb_cls'])
            if 'img_size' in ck_args and isinstance(ck_args['img_size'], (list, tuple)) and len(ck_args['img_size']) == 2:
                img_size = list(ck_args['img_size'])
    except Exception:
        pass

    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size[::-1])

    model_dict = OrderedDict()
    pattern = re.compile('module.')

    # Prefer EMA weights if present, else fall back to 'model'
    state = None
    if 'state_dict_ema' in ckpt and ckpt['state_dict_ema'] is not None:
        state = ckpt['state_dict_ema']
    elif 'model' in ckpt and ckpt['model'] is not None:
        state = ckpt['model']
    else:
        # try root-level state_dict
        for k in ['state_dict', 'model_state_dict']:
            if k in ckpt:
                state = ckpt[k]
                break
    if state is None:
        raise KeyError('No model weights found in checkpoint')

    for k, v in state.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f'Model loaded successfully on {device} | nb_cls={nb_cls} | img_size={img_size}')
    return model, device, nb_cls, img_size


def preprocess_image(image_path, img_size, threshold=None):
    """Load and preprocess a single image to match training: resize height, preserve aspect, pad width, scale to [0,1].
    Optional binarization threshold in [0,1]. Returns tensor (1,1,H,W).
    """
    try:
        img = Image.open(image_path).convert('L')
        max_w, max_h = int(img_size[0]), int(img_size[1])

        # Resize to target height, preserve aspect, then cap width to max_w
        w0, h0 = img.size
        new_h = max_h
        new_w = min(max_w, max(1, int(round(w0 * (new_h / float(h0))))))
        img = img.resize((new_w, new_h))

        # To numpy float32 in [0,1]
        ar = np.array(img).astype(np.float32) / 255.0
        # Optional binarization
        if threshold is not None:
            ar = (ar >= float(threshold)).astype(np.float32)

        # Pad to max_w with white (1.0) on the right
        if new_w < max_w:
            pad = np.ones((new_h, max_w - new_w), dtype=np.float32)
            ar = np.concatenate([ar, pad], axis=1)

        # (H,W) -> (1,1,H,W)
        t = torch.from_numpy(ar)[None, None, ...]
        print(f'Image loaded: {image_path}, shape: {t.shape}, dtype={t.dtype}, range=({t.min().item():.3f},{t.max().item():.3f})')
        return t
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        return None

def load_ground_truth(text_path):
    """Load ground truth text"""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        print(f'Ground truth loaded: "{text}"')
        return text
    except Exception as e:
        print(f'Error loading text {text_path}: {e}')
        return ""


def create_converter():
    """Create character converter with Vietnamese character set"""
    # Vietnamese character set (should match training)
    char_string = ('abcdefghijklmnopqrstuvwxyz'
                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                   '0123456789'
                   '.,!?;: "#&\'()*+-/%=<>@[]^_`{|}~'
                   'àáảãạăằắẳẵặâầấẩẫậ'
                   'èéẻẽẹêềếểễệ'
                   'ìíỉĩị'
                   'òóỏõọôồốổỗộơờớởỡợ'
                   'ùúủũụưừứửữự'
                   'ỳýỷỹỵ'
                   'đ'
                   'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ'
                   'ÈÉẺẼẸÊỀẾỂỄỆ'
                   'ÌÍỈĨỊ'
                   'ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ'
                   'ÙÚỦŨỤƯỪỨỬỮỰ'
                   'ỲÝỶỸỴ'
                   'Đ')

    ralph = {idx: char for idx, char in enumerate(char_string)}
    converter = utils.CTCLabelConverter(ralph.values())

    print(f'Character set size (incl. [blank]): {len(converter.character)}')
    return converter


def predict_text(model, image, converter, device, use_tone_head=True, tone_tau_v=0.5, tone_kappa=0.3):
    """Predict text with optional tone composition using the tone head.
    image: (1,1,H,W). Returns predicted string.
    """
    with torch.no_grad():
        image = image.to(device)

        outputs = model(image, use_masking=False)
        if isinstance(outputs, (list, tuple)):
            base_logits, tone_logits = outputs
        else:
            base_logits, tone_logits = outputs, None
        base_logits = base_logits.float()  # (B=1, T, C)

        # Greedy base decode (CTC)
        T_len = base_logits.size(1)
        preds_size = torch.IntTensor([T_len])
        logp_tbc = base_logits.permute(1, 0, 2).log_softmax(2)
        _, preds_index = logp_tbc.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        base_str = converter.decode(preds_index.data, preds_size.data)
        base_str = base_str[0] if base_str else ""

        # If no tone head or disabled, return base string
        if (tone_logits is None) or (not use_tone_head):
            return base_str

        # Tone-aware composition from spans
        base_probs = F.softmax(base_logits, dim=-1)  # (1,T,C)
        tone_probs = F.softmax(tone_logits, dim=-1)  # (1,T,6)

        # Vowel indices and mask
        vowel_idxs = utils.vowel_indices_from_converter(converter)
        C = base_probs.size(-1)
        vowel_mask = torch.zeros(C, device=base_probs.device)
        if len(vowel_idxs) > 0:
            vowel_mask.scatter_(0, torch.tensor(vowel_idxs, device=base_probs.device, dtype=torch.long), 1.0)

        # Framewise labels
        frame_labels = base_logits.argmax(dim=-1)  # (1,T)
        b = 0
        T = T_len

        # Build spans (c_idx, start, end)
        spans = []
        last_c = 0
        start = None
        for t in range(T):
            c = int(frame_labels[b, t].item())
            if c == 0:
                if start is not None:
                    spans.append((last_c, start, t))
                    start = None
                last_c = 0
                continue
            if start is None:
                start = t
                last_c = c
            elif c != last_c:
                spans.append((last_c, start, t))
                start = t
                last_c = c
        if start is not None:
            spans.append((last_c, start, T))

        # Gate per frame: vowel mass >= tau_v
        v_scores = (base_probs[b, :, :] * vowel_mask.view(1, -1)).sum(dim=-1)
        gate = (v_scores >= float(tone_tau_v)).float()

        # Compose final string
        chars = []
        for c_idx, s, e in spans:
            if c_idx <= 0 or c_idx >= len(converter.character):
                continue
            ch = converter.character[c_idx]
            if utils.is_vietnamese_vowel(ch):
                g_span = gate[s:e]
                denom = g_span.sum()
                if float(denom.item()) < 1e-6:
                    g_span = torch.ones_like(g_span)
                    denom = g_span.sum()
                avg = (tone_probs[b, s:e, :] * g_span.view(-1, 1)).sum(dim=0) / denom
                best_tone = int(avg[1:].argmax().item()) + 1  # 1..5
                margin = float(avg[best_tone] - avg[0])
                tone_id = best_tone if margin >= float(tone_kappa) else 0
                ch = utils.apply_tone_to_char(ch, tone_id)
            chars.append(ch)

        return ''.join(chars)


def calculate_metrics(predicted, ground_truth):
    """Calculate CER and WER"""
    import editdistance

    # Character Error Rate (CER)
    if len(ground_truth) == 0:
        cer = 1.0 if len(predicted) > 0 else 0.0
    else:
        cer = editdistance.eval(predicted, ground_truth) / len(ground_truth)

    # Word Error Rate (WER)
    pred_words = predicted.split()
    gt_words = ground_truth.split()

    if len(gt_words) == 0:
        wer = 1.0 if len(pred_words) > 0 else 0.0
    else:
        wer = editdistance.eval(pred_words, gt_words) / len(gt_words)

    return cer, wer


def main():
    parser = argparse.ArgumentParser(
        description='Quick HTR inference on single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--text', type=str,
                        help='Path to ground truth text file (optional)')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, nargs=2,
                        default=[512, 64], help='Image size [width, height]')
    parser.add_argument('--nb_cls', type=int, default=228,
                        help='Number of character classes')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Optional binarization threshold in [0,1] (e.g., 0.7). If unset, no binarization.')
    parser.add_argument('--use_tone_head', action='store_true', default=True,
                        help='Enable tone composition from tone head')
    parser.add_argument('--tone_tau_v', type=float, default=0.5,
                        help='Vowel gate threshold on base probs')
    parser.add_argument('--tone_kappa', type=float, default=0.3,
                        help='Margin over NONE to accept a tone')

    args = parser.parse_args()

    print("=" * 60)
    print("HTR-VT Quick Inference")
    print("=" * 60)

    # Load model (may override nb_cls/img_size from checkpoint args)
    model, device, args.nb_cls, args.img_size = load_model(args.checkpoint, args.nb_cls, args.img_size)

    # Create converter
    converter = create_converter()

    # Check character count compatibility
    if len(converter.character) != args.nb_cls:
        print(
            f"WARNING: Character count mismatch! Converter: {len(converter.character)}, Model: {args.nb_cls}")

    # Load and preprocess image
    image = preprocess_image(args.image, args.img_size, threshold=args.threshold)
    if image is None:
        return

    # Load ground truth if provided
    ground_truth = ""
    if args.text and os.path.exists(args.text):
        ground_truth = load_ground_truth(args.text)

    # Predict
    print("\nRunning inference...")
    predicted = predict_text(
        model,
        image,
        converter,
        device,
        use_tone_head=args.use_tone_head,
        tone_tau_v=args.tone_tau_v,
        tone_kappa=args.tone_kappa,
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Predicted: \"{predicted}\"")

    if ground_truth:
        print(f"Ground Truth: \"{ground_truth}\"")

        # Calculate metrics
        cer, wer = calculate_metrics(predicted, ground_truth)
        print(f"\nMetrics:")
        print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
        print(f"WER: {wer:.4f} ({wer*100:.2f}%)")

        # Show character-level comparison
        print(f"\nComparison:")
        print(f"Predicted length: {len(predicted)} characters")
        print(f"Ground truth length: {len(ground_truth)} characters")

        if predicted == ground_truth:
            print("✅ Perfect match!")
        else:
            print("❌ Mismatch detected")
    else:
        print("(No ground truth provided)")

    print("=" * 60)


if __name__ == '__main__':
    main()
