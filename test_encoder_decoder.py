import torch
import torch.utils.data
from collections import OrderedDict
import re
import os
import json
import argparse
import editdistance

from utils import utils
from utils.encoder_decoder_tokenizer import EncoderDecoderTokenizer
from data import dataset
from model.HTR_EncoderDecoder import create_encoder_decoder_model


def load_model_and_tokenizer(checkpoint_path, device='cuda'):
    """
    Load trained encoder-decoder model and tokenizer.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        tokenizer: Tokenizer instance
        args: Training arguments
    """
    print(f'Loading checkpoint from {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Get tokenizer info and training args
    tokenizer_info = ckpt['tokenizer_info']
    train_args = ckpt.get('args', {})
    
    # Reconstruct alphabet
    character = ''.join(tokenizer_info['character'][4:])  # Remove special tokens
    tokenizer = EncoderDecoderTokenizer(character)
    
    # Create model
    model = create_encoder_decoder_model(
        vocab_size=tokenizer_info['vocab_size'],
        img_size=train_args.get('img_size', [64, 512])[::-1],
        max_seq_len=256
    )
    
    # Load state dict
    if 'state_dict_ema' in ckpt:
        state_dict = ckpt['state_dict_ema']
    else:
        state_dict = ckpt['model']
    
    # Handle module prefix
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v
    
    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, train_args


def test_single_image(model, tokenizer, image_path, device='cuda', max_length=256, method='beam_search'):
    """
    Test model on a single image.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        image_path: Path to image file
        device: Device for computation
        max_length: Maximum generation length
        method: Decoding method ('greedy' or 'beam_search')
    
    Returns:
        predicted_text: Predicted text string
        confidence_score: Confidence score (for beam search)
    """
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to model input size (typically 64x512)
    transform = transforms.Compose([
        transforms.Resize((64, 512)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, 64, 512]
    
    with torch.no_grad():
        if method == 'beam_search':
            sequences, scores = model.generate(
                image_tensor,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_len=max_length,
                method='beam_search',
                beam_size=5
            )
            # Take best beam
            best_sequence = sequences[0:1]
            best_score = scores[0].item() if scores is not None else 0.0
        else:
            sequences, _ = model.generate(
                image_tensor,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_len=max_length,
                method='greedy'
            )
            best_sequence = sequences
            best_score = 0.0
        
        # Decode to text
        predicted_texts = tokenizer.decode(best_sequence)
        predicted_text = predicted_texts[0]
    
    return predicted_text, best_score


def test_dataset(model, tokenizer, test_loader, device='cuda', max_length=256):
    """
    Test model on a dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        test_loader: DataLoader for test set
        device: Device for computation
        max_length: Maximum generation length
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    
    print('Running inference on test set...')
    with torch.no_grad():
        for i, (images, texts) in enumerate(test_loader):
            if i % 100 == 0:
                print(f'Processed {i}/{len(test_loader)} batches')
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate predictions
            if batch_size == 1:
                # Use beam search for single samples
                try:
                    pred_sequences, _ = model.generate(
                        images,
                        sos_token_id=tokenizer.sos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_len=max_length,
                        method='beam_search',
                        beam_size=5
                    )
                    pred_sequences = pred_sequences[0:1]
                except:
                    # Fallback to greedy
                    pred_sequences, _ = model.generate(
                        images,
                        sos_token_id=tokenizer.sos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_len=max_length,
                        method='greedy'
                    )
            else:
                # Use greedy for batches
                pred_sequences, _ = model.generate(
                    images,
                    sos_token_id=tokenizer.sos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_len=max_length,
                    method='greedy'
                )
            
            # Decode predictions
            predicted_texts = tokenizer.decode(pred_sequences)
            
            # Compute metrics
            for pred_text, gt_text in zip(predicted_texts, texts):
                # CER
                cer = editdistance.eval(pred_text, gt_text) / max(len(gt_text), 1)
                total_cer += cer
                
                # WER
                pred_words = utils.format_string_for_wer(pred_text).split()
                gt_words = utils.format_string_for_wer(gt_text).split()
                wer = editdistance.eval(pred_words, gt_words) / max(len(gt_words), 1)
                total_wer += wer
                
                num_samples += 1
            
            all_predictions.extend(predicted_texts)
            all_ground_truths.extend(texts)
    
    avg_cer = total_cer / num_samples if num_samples > 0 else 1.0
    avg_wer = total_wer / num_samples if num_samples > 0 else 1.0
    
    results = {
        'num_samples': num_samples,
        'cer': avg_cer,
        'wer': avg_wer,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test HTR Encoder-Decoder Model')
    
    # Model and data arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data-list', type=str, required=True,
                       help='Path to test data list file')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+',
                       help='Input image size')
    
    # Inference arguments
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum generation length')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for testing')
    parser.add_argument('--method', type=str, default='beam_search',
                       choices=['greedy', 'beam_search'],
                       help='Decoding method')
    
    # Single image testing
    parser.add_argument('--single-image', type=str, default=None,
                       help='Path to single image for testing')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, train_args = load_model_and_tokenizer(args.checkpoint, device)
    
    print(f'Loaded model with vocab size: {tokenizer.vocab_size}')
    print(f'Special tokens - SOS: {tokenizer.sos_token_id}, EOS: {tokenizer.eos_token_id}, PAD: {tokenizer.pad_token_id}')
    
    # Test single image if provided
    if args.single_image:
        print(f'\nTesting single image: {args.single_image}')
        predicted_text, confidence = test_single_image(
            model, tokenizer, args.single_image, device, args.max_length, args.method
        )
        print(f'Predicted text: "{predicted_text}"')
        if args.method == 'beam_search':
            print(f'Confidence score: {confidence:.4f}')
        return
    
    # Test on dataset
    print(f'\nTesting on dataset: {args.test_data_list}')
    
    # Get alphabet from training data (from tokenizer)
    alphabet = ''.join(tokenizer.character[4:])  # Remove special tokens
    
    # Load test dataset
    test_dataset = dataset.myLoadDS(
        args.test_data_list, 
        args.data_path, 
        args.img_size,
        ralph=alphabet,
        fmin=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False
    )
    
    print(f'Test dataset size: {len(test_dataset)}')
    
    # Run evaluation
    results = test_dataset(model, tokenizer, test_loader, device, args.max_length)
    
    # Print results
    print(f'\nTest Results:')
    print(f'Number of samples: {results["num_samples"]}')
    print(f'Character Error Rate (CER): {results["cer"]:.4f}')
    print(f'Word Error Rate (WER): {results["wer"]:.4f}')
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, 'test_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'num_samples': results['num_samples'],
            'cer': results['cer'],
            'wer': results['wer'],
            'checkpoint': args.checkpoint,
            'test_data': args.test_data_list,
            'method': args.method,
            'max_length': args.max_length
        }, f, indent=2)
    
    # Save predictions
    pred_file = os.path.join(args.output_dir, 'predictions.txt')
    with open(pred_file, 'w', encoding='utf-8') as f:
        for pred, gt in zip(results['predictions'], results['ground_truths']):
            f.write(f'GT: {gt}\n')
            f.write(f'Pred: {pred}\n')
            f.write('-' * 50 + '\n')
    
    print(f'\nResults saved to {output_file}')
    print(f'Predictions saved to {pred_file}')
    
    # Show some sample predictions
    print('\nSample predictions:')
    for i in range(min(5, len(results['predictions']))):
        print(f'GT:   {results["ground_truths"][i]}')
        print(f'Pred: {results["predictions"][i]}')
        print()


if __name__ == '__main__':
    main()
