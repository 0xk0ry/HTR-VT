"""
Example script demonstrating the Encoder-Decoder HTR model.
This shows how to use the new model for training and inference.
"""

import torch
import argparse
import numpy as np
from PIL import Image

from model.HTR_EncoderDecoder import create_encoder_decoder_model
from utils.encoder_decoder_tokenizer import EncoderDecoderTokenizer
from utils.encoder_decoder_utils import compute_encoder_decoder_loss


def create_dummy_data(batch_size=2, img_size=(64, 512), vocab_size=80):
    """Create dummy data for testing."""
    # Create dummy images
    images = torch.randn(batch_size, 1, img_size[0], img_size[1])
    
    # Create dummy texts
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'
    texts = []
    for _ in range(batch_size):
        length = np.random.randint(5, 20)
        text = ''.join(np.random.choice(list(alphabet), length))
        texts.append(text)
    
    return images, texts, alphabet


def demo_training():
    """Demonstrate training procedure."""
    print("=== Training Demo ===")
    
    # Create dummy data
    images, texts, alphabet = create_dummy_data(batch_size=4)
    print(f"Sample texts: {texts}")
    
    # Create tokenizer
    tokenizer = EncoderDecoderTokenizer(alphabet)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens - SOS: {tokenizer.sos_token_id}, EOS: {tokenizer.eos_token_id}")
    
    # Create model
    model = create_encoder_decoder_model(
        vocab_size=tokenizer.vocab_size,
        img_size=[512, 64],  # [W, H]
        max_seq_len=128
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Compute loss
    loss, logits, targets = compute_encoder_decoder_loss(
        model, images, texts, tokenizer, max_length=64
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training step completed!")


def demo_inference():
    """Demonstrate inference procedure."""
    print("\n=== Inference Demo ===")
    
    # Create dummy data
    images, texts, alphabet = create_dummy_data(batch_size=1)
    
    # Create tokenizer
    tokenizer = EncoderDecoderTokenizer(alphabet)
    
    # Create model
    model = create_encoder_decoder_model(
        vocab_size=tokenizer.vocab_size,
        img_size=[512, 64],
        max_seq_len=128
    )
    
    model.eval()
    
    print(f"Ground truth: '{texts[0]}'")
    
    # Greedy decoding
    with torch.no_grad():
        sequences, _ = model.generate(
            images,
            sos_token_id=tokenizer.sos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            method='greedy',
            max_len=64
        )
    
    predicted_texts = tokenizer.decode(sequences)
    print(f"Greedy prediction: '{predicted_texts[0]}'")
    
    # Beam search decoding
    with torch.no_grad():
        sequences, scores = model.generate(
            images,
            sos_token_id=tokenizer.sos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            method='beam_search',
            beam_size=3,
            max_len=64
        )
    
    beam_texts = tokenizer.decode(sequences)
    print(f"Beam search predictions:")
    for i, (text, score) in enumerate(zip(beam_texts, scores)):
        print(f"  Beam {i+1}: '{text}' (score: {score:.4f})")


def demo_tokenizer():
    """Demonstrate tokenizer functionality."""
    print("\n=== Tokenizer Demo ===")
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    tokenizer = EncoderDecoderTokenizer(alphabet)
    
    print(f"Alphabet: {alphabet}")
    print(f"Vocabulary: {tokenizer.character}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Sample texts
    texts = ['hello world', 'test 123', 'encoder decoder']
    
    # Encode for training
    tgt_input, tgt_output, lengths = tokenizer.encode_for_training(texts, max_length=20)
    
    print(f"\nSample encoding:")
    for i, text in enumerate(texts):
        print(f"Text: '{text}'")
        print(f"  Input:  {tgt_input[i].tolist()}")
        print(f"  Output: {tgt_output[i].tolist()}")
        print(f"  Length: {lengths[i].item()}")
    
    # Decode back
    decoded_texts = tokenizer.decode(tgt_output)
    print(f"\nDecoded texts: {decoded_texts}")


def demo_encoder_output():
    """Demonstrate encoder output format."""
    print("\n=== Encoder Output Demo ===")
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    tokenizer = EncoderDecoderTokenizer(alphabet)
    
    # Create model
    model = create_encoder_decoder_model(
        vocab_size=tokenizer.vocab_size,
        img_size=[512, 64],
        max_seq_len=128
    )
    
    # Create sample image
    batch_size = 2
    images = torch.randn(batch_size, 1, 64, 512)
    
    print(f"Input images shape: {images.shape}")
    
    # Get encoder output
    model.eval()
    with torch.no_grad():
        memory = model.encode(images)
    
    print(f"Encoder output (memory) shape: {memory.shape}")
    print(f"Format: [sequence_length, batch_size, embedding_dim]")
    
    # This matches the expected input format for the transformer decoder
    seq_len, batch_size, embed_dim = memory.shape
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Embedding dimension: {embed_dim}")


def main():
    parser = argparse.ArgumentParser(description='Encoder-Decoder HTR Model Demo')
    parser.add_argument('--demo', type=str, choices=['training', 'inference', 'tokenizer', 'encoder', 'all'],
                       default='all', help='Which demo to run')
    
    args = parser.parse_args()
    
    print("HTR Encoder-Decoder Model Demo")
    print("=" * 50)
    
    if args.demo in ['training', 'all']:
        demo_training()
    
    if args.demo in ['inference', 'all']:
        demo_inference()
    
    if args.demo in ['tokenizer', 'all']:
        demo_tokenizer()
    
    if args.demo in ['encoder', 'all']:
        demo_encoder_output()
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()
