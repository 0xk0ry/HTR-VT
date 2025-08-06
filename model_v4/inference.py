import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import os
import re
from collections import OrderedDict

from model import HTR_VT
from data.dataset import get_images, merge_base_and_diacritic
from utils import utils


def load_model(checkpoint_path, nb_cls, img_size):
    """Load trained multi-task model"""
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size[::-1])
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    
    # Handle both 'state_dict_ema' and 'model' keys
    state_dict_key = 'state_dict_ema' if 'state_dict_ema' in ckpt else 'model'
    
    for k, v in ckpt[state_dict_key].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v
    
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    return model


def preprocess_image(image_path, img_size):
    """Preprocess single image for inference"""
    img_data = get_images(image_path, img_size[0], img_size[1])
    img_data = img_data.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_data).unsqueeze(0).float()
    return img_tensor


def inference_single_image(model, image_path, converter, img_size, device='cuda'):
    """
    Perform inference on a single image
    Returns: base prediction, diacritic prediction, merged Vietnamese text
    """
    # Preprocess image
    img_tensor = preprocess_image(image_path, img_size)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Get model predictions
        base_logits, diacritic_logits = model(img_tensor)
        
        # Decode base characters using CTC
        base_logits_ctc = base_logits.permute(1, 0, 2).log_softmax(2)
        _, preds_index = base_logits_ctc.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.IntTensor([base_logits.size(1)])
        base_pred = converter.decode(preds_index.data, preds_size.data)[0]
        
        # Decode diacritic predictions
        diacritic_pred = diacritic_logits.argmax(dim=-1).cpu().numpy()[0]
        
        # Merge base and diacritic to get final Vietnamese text
        # Truncate diacritic prediction to match base prediction length
        diacritic_pred = diacritic_pred[:len(base_pred)]
        merged_text = merge_base_and_diacritic(base_pred, diacritic_pred)
    
    return base_pred, diacritic_pred.tolist(), merged_text


def main():
    parser = argparse.ArgumentParser(description='Multi-task HTR-VT Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--nb-cls', type=int, default=84,
                        help='Number of character classes')
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+',
                        help='Image size [width, height]')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create base alphabet (same as in dataset)
    base_alphabet = (
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '.,!?;: "#&\'()*+-/%=<>@[]^_`{|}~'
        'ăâêôơưđ'
        'ĂÂÊÔƠƯĐ'
    )
    
    # Create converter
    converter = utils.CTCLabelConverter(base_alphabet)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.nb_cls, args.img_size)
    model = model.to(args.device)
    
    # Perform inference
    print(f"Processing image: {args.image}")
    base_pred, diacritic_pred, merged_text = inference_single_image(
        model, args.image, converter, args.img_size, args.device
    )
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Base characters: {base_pred}")
    print(f"Diacritic classes: {diacritic_pred}")
    print(f"Final Vietnamese text: {merged_text}")
    print("="*50)


if __name__ == '__main__':
    main()
