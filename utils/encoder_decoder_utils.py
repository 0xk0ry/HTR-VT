import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_encoder_decoder_loss(model, image, texts, tokenizer, max_length=None, label_smoothing=0.1):
    """
    Compute cross-entropy loss for encoder-decoder model.
    
    Args:
        model: HTR_EncoderDecoder model
        image: Input images [B, C, H, W]
        texts: List of text strings (ground truth)
        tokenizer: EncoderDecoderTokenizer instance
        max_length: Maximum sequence length
        label_smoothing: Label smoothing factor
    
    Returns:
        loss: CrossEntropyLoss
        logits: Model output logits [T, B, vocab_size]
        targets: Target tokens [B, T]
    """
    batch_size = image.size(0)
    
    # Encode texts to input/output sequences
    tgt_input, tgt_output, lengths = tokenizer.encode_for_training(texts, max_length)
    
    # Convert to [T, B] format for transformer
    tgt_input_t = tgt_input.transpose(0, 1)  # [T, B]
    tgt_output_t = tgt_output.transpose(0, 1)  # [T, B]
    
    # Create padding mask for target sequence
    tgt_key_padding_mask = tokenizer.create_padding_mask(tgt_input)
    
    # Forward pass
    logits = model(image, tgt=tgt_input_t, tgt_key_padding_mask=tgt_key_padding_mask)
    
    # Prepare loss computation
    # logits: [T, B, vocab_size], tgt_output_t: [T, B]
    T, B, vocab_size = logits.shape
    
    # Flatten for loss computation
    logits_flat = logits.view(-1, vocab_size)  # [T*B, vocab_size]
    targets_flat = tgt_output_t.view(-1)  # [T*B]
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=label_smoothing
    )
    
    # Compute loss
    loss = criterion(logits_flat, targets_flat)
    
    return loss, logits, tgt_output


def compute_encoder_decoder_loss_with_mask(model, image, texts, tokenizer, max_length=None, label_smoothing=0.1):
    """
    Compute cross-entropy loss with explicit masking (alternative implementation).
    
    Args:
        model: HTR_EncoderDecoder model
        image: Input images [B, C, H, W]
        texts: List of text strings (ground truth)
        tokenizer: EncoderDecoderTokenizer instance
        max_length: Maximum sequence length
        label_smoothing: Label smoothing factor
    
    Returns:
        loss: CrossEntropyLoss
        logits: Model output logits [T, B, vocab_size]
        targets: Target tokens [B, T]
    """
    batch_size = image.size(0)
    
    # Encode texts to input/output sequences
    tgt_input, tgt_output, lengths = tokenizer.encode_for_training(texts, max_length)
    
    # Convert to [T, B] format for transformer
    tgt_input_t = tgt_input.transpose(0, 1)  # [T, B]
    tgt_output_t = tgt_output.transpose(0, 1)  # [T, B]
    
    # Create padding mask for target sequence
    tgt_key_padding_mask = tokenizer.create_padding_mask(tgt_input)
    
    # Forward pass
    logits = model(image, tgt=tgt_input_t, tgt_key_padding_mask=tgt_key_padding_mask)
    
    # Create mask for valid positions (not padding)
    valid_mask = (tgt_output_t != tokenizer.pad_token_id)  # [T, B]
    
    # Compute loss only on valid positions
    T, B, vocab_size = logits.shape
    
    # Apply mask
    valid_logits = logits[valid_mask]  # [num_valid, vocab_size]
    valid_targets = tgt_output_t[valid_mask]  # [num_valid]
    
    if valid_logits.size(0) > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = criterion(valid_logits, valid_targets)
    else:
        # Fallback if no valid tokens
        loss = torch.tensor(0.0, requires_grad=True, device=image.device)
    
    return loss, logits, tgt_output


def evaluate_encoder_decoder(model, eval_loader, tokenizer, device, max_length=None):
    """
    Evaluate encoder-decoder model on validation/test set.
    
    Args:
        model: HTR_EncoderDecoder model
        eval_loader: DataLoader for evaluation
        tokenizer: EncoderDecoderTokenizer instance
        device: Device for computation
        max_length: Maximum generation length
    
    Returns:
        avg_loss: Average validation loss
        predictions: List of predicted texts
        ground_truths: List of ground truth texts
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, texts in eval_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Compute loss
            loss, _, _ = compute_encoder_decoder_loss(model, images, texts, tokenizer, max_length)
            total_loss += loss.item()
            num_batches += 1
            
            # Generate predictions
            if batch_size == 1:
                # Use beam search for single samples
                pred_sequences, _ = model.generate(
                    images,
                    sos_token_id=tokenizer.sos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_len=max_length,
                    method='beam_search',
                    beam_size=5
                )
                # Take best beam
                pred_sequences = pred_sequences[0:1]
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
            
            all_predictions.extend(predicted_texts)
            all_ground_truths.extend(texts)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, all_predictions, all_ground_truths


def create_tgt_mask(seq_len, device):
    """
    Create subsequent mask for target sequence (causal mask).
    
    Args:
        seq_len: Sequence length
        device: Device for tensor
    
    Returns:
        mask: [seq_len, seq_len] causal mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float('-inf')).to(device)
    return mask


def warmup_cosine_schedule(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    """
    Learning rate schedule with warmup and cosine annealing.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
    
    Returns:
        lr: Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        lr = max_lr * step / warmup_steps
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return lr


def update_learning_rate(optimizer, lr):
    """Update learning rate for optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
