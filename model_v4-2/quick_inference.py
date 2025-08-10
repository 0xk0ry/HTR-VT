import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import argparse

import os
import re
import json
from utils import utils
from utils import option
from model import HTR_VT
from collections import OrderedDict


def load_model(checkpoint_path, nb_cls, img_size):
    """Load the trained model from checkpoint"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size[::-1])

    print(f'Loading model from: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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

    print(f'Model loaded successfully on {device}')
    return model, device


def preprocess_image(image_path, img_size, threshold=0.7):
    """Load and preprocess a single image, with thresholding"""
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Image transforms (same as training)
        transform = transforms.Compose([
            transforms.Resize((img_size[1], img_size[0])),  # height, width
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

        image_original = transform(image).unsqueeze(0)  # Add batch dimension

        # Thresholding: set background (< x) to white, foreground (>= x) to black
        # Undo normalization for thresholding
        if threshold is not None:
            image_unnorm = image_original * 0.5 + 0.5  # Back to [0, 1]
            mask = image_unnorm < threshold
            image_unnorm[mask] = 0.0  # White
            image_unnorm[~mask] = 1.0  # Black
        else:
            for tmp in range (0, 11):
                image_unnorm = image_original * 0.5 + 0.5  # Back to [0, 1]
                threshold = tmp * 0.1
                mask = image_unnorm < threshold
                image_unnorm[mask] = 0.0  # White
                image_unnorm[~mask] = 1.0  # Black

                from torchvision.transforms import ToPILImage
                # Remove batch dimension for saving
                img_to_save = image_unnorm.squeeze(0)
                pil_img = ToPILImage()(img_to_save)
                pil_img.save(f'preprocessed_image_{threshold:.2f}.png')
                print(f"Preprocessed image saved to: preprocessed_image_{threshold:.2f}.png")

        # Re-normalize to [-1, 1]
        image = (image_unnorm - 0.5) / 0.5

        print(f'Image loaded: {image_path}, shape: {image.shape}')
        return image

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

    print(f'Character set size: {len(converter.character)}')
    return converter


def predict_text(model, image, converter, device):
    """Predict text from image"""
    with torch.no_grad():
        image = image.to(device)

        # Get model prediction
        preds = model(image)
        preds = preds.float()

        # Decode prediction
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)

        # Convert to text
        pred_str = converter.decode(preds_index.data, preds_size.data)

        return pred_str[0] if pred_str else ""


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

    args = parser.parse_args()

    print("=" * 60)
    print("HTR-VT Quick Inference")
    print("=" * 60)

    # Load model
    model, device = load_model(args.checkpoint, args.nb_cls, args.img_size)

    # Create converter
    converter = create_converter()

    # Check character count compatibility
    if len(converter.character) != args.nb_cls:
        print(
            f"WARNING: Character count mismatch! Converter: {len(converter.character)}, Model: {args.nb_cls}")

    # Load and preprocess image
    image = preprocess_image(args.image, args.img_size)
    if image is None:
        return

    # Load ground truth if provided
    ground_truth = ""
    if args.text and os.path.exists(args.text):
        ground_truth = load_ground_truth(args.text)

    # Predict
    print("\nRunning inference...")
    predicted = predict_text(model, image, converter, device)

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
