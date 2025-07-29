import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import editdistance
from utils import utils
from utils import sam
from utils import option
from utils.encoder_decoder_tokenizer import EncoderDecoderTokenizer
from utils.encoder_decoder_utils import (
    compute_encoder_decoder_loss,
    evaluate_encoder_decoder,
    warmup_cosine_schedule,
    update_learning_rate
)
from data import dataset
from model.HTR_EncoderDecoder import create_encoder_decoder_model
from functools import partial


def get_alphabet_from_data(data_list, data_path):
    """Extract alphabet from ground truth .txt files."""
    alphabet = set()
    
    print(f"DEBUG: Reading image list from: {data_list}")
    print(f"DEBUG: Looking for .txt files in: {data_path}")
    
    try:
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"DEBUG: Found {len(lines)} image entries in .ln file")
        
        # Debug first few lines
        for i, line in enumerate(lines[:3]):
            print(f"DEBUG: Line {i}: {repr(line.strip())}")
        
        processed_count = 0
        error_count = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Extract image filename (remove .png extension, add .txt)
            # Handle both "image.png" and "image.png\ttext" formats
            parts = line.split('\t')
            image_filename = parts[0].strip()
            
            # Convert image filename to txt filename
            if image_filename.endswith('.png'):
                txt_filename = image_filename[:-4] + '.txt'
            else:
                txt_filename = image_filename + '.txt'
            
            txt_path = os.path.join(data_path, txt_filename)
            
            try:
                # Read ground truth text from .txt file
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    ground_truth_text = txt_file.read().strip()
                    alphabet.update(ground_truth_text)
                    processed_count += 1
                    
                    if line_num < 3:  # Debug first few
                        print(f"DEBUG: {image_filename} -> {txt_filename}: '{ground_truth_text}'")
                        
            except FileNotFoundError:
                error_count += 1
                if error_count <= 5:  # Show first few errors
                    print(f"DEBUG: Missing txt file: {txt_path}")
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"DEBUG: Error reading {txt_path}: {e}")
        
        print(f"DEBUG: Processed {processed_count} files, {error_count} errors")
        
        # Sort to ensure consistent ordering
        alphabet_str = ''.join(sorted(alphabet))
        print(f"DEBUG: Extracted alphabet ({len(alphabet_str)} chars): {alphabet_str[:100]}...")
        
        if len(alphabet_str) == 0:
            print("ERROR: No alphabet extracted! Check data paths and file structure")
            print(f"Expected structure: {data_path}/image_name.txt containing ground truth text")
            raise ValueError(f"No characters found in ground truth files")
        
        return alphabet_str
        
    except FileNotFoundError:
        print(f"ERROR: List file not found: {data_list}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to extract alphabet: {e}")
        raise


def validation_encoder_decoder(model, criterion, evaluation_loader, tokenizer, device, max_length=None):
    """Validation function for encoder-decoder model."""
    model.eval()
    
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, texts in evaluation_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Compute loss
            loss, _, _ = compute_encoder_decoder_loss(model, images, texts, tokenizer, max_length)
            total_loss += loss.item() * batch_size
            
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
                        beam_size=3
                    )
                    pred_sequences = pred_sequences[0:1]
                except:
                    # Fallback to greedy if beam search fails
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
            
            # Compute CER and WER
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
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 1.0
    avg_wer = total_wer / num_samples if num_samples > 0 else 1.0
    
    return avg_loss, avg_cer, avg_wer, all_predictions, all_ground_truths


def main():
    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name + '_encoder_decoder')
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    # Extract alphabet from training data
    alphabet = get_alphabet_from_data(args.train_data_list, args.data_path)
    logger.info(f'Alphabet size: {len(alphabet)}')
    logger.info(f'Alphabet: {alphabet}')
    
    # Create tokenizer
    tokenizer = EncoderDecoderTokenizer(alphabet)
    vocab_info = tokenizer.get_vocab_info()
    logger.info(f'Vocabulary size: {vocab_info["vocab_size"]}')
    logger.info(f'First 10 characters: {vocab_info["character"][:10]}')
    
    # Debug: Test encoding/decoding
    test_text = ["hello world", "test"]
    test_input, test_output, test_lengths = tokenizer.encode_for_training(test_text)
    test_decoded = tokenizer.decode(test_output)
    logger.info(f'Tokenizer test - Original: {test_text}')
    logger.info(f'Tokenizer test - Decoded: {test_decoded}')
    
    # Save tokenizer info
    with open(os.path.join(args.save_dir, 'tokenizer_info.json'), 'w') as f:
        json.dump(vocab_info, f, indent=2)

    # Create model
    model = create_encoder_decoder_model(
        vocab_size=vocab_info['vocab_size'],
        img_size=args.img_size[::-1],
        max_seq_len=256
    )

    # Verify model vocab size matches tokenizer
    model_vocab_size = model.decoder.output_projection.out_features
    logger.info(f'Model vocab size: {model_vocab_size}')
    logger.info(f'Tokenizer vocab size: {vocab_info["vocab_size"]}')
    if model_vocab_size != vocab_info['vocab_size']:
        logger.error(f'MISMATCH: Model vocab size ({model_vocab_size}) != Tokenizer vocab size ({vocab_info["vocab_size"]})')
        raise ValueError("Vocabulary size mismatch!")

    total_param = sum(p.numel() for p in model.parameters())
    logger.info(f'Total parameters: {total_param}')

    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size, 
                                     ralph=alphabet)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=partial(dataset.SameTrCollate, args=args),
        drop_last=True
    )

    logger.info('Loading validation loader...')
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, 
                                   ralph=alphabet, fmin=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False
    )

    # SAM optimizer
    base_optimizer = torch.optim.AdamW
    optimizer = sam.SAM(model.parameters(), base_optimizer, lr=args.max_lr, 
                        weight_decay=args.weight_decay)

    # Training loop
    best_cer = float('inf')
    best_wer = float('inf')
    train_loss = 0.0
    nb_iter = 0
    
    train_iter = iter(train_loader)

    logger.info('Starting training...')
    while nb_iter < args.total_iter:
        nb_iter += 1
        
        # Update learning rate
        current_lr = warmup_cosine_schedule(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr
        )
        update_learning_rate(optimizer, current_lr)

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        image = batch[0].cuda()
        texts = batch[1]
        batch_size = image.size(0)

        # First forward-backward pass
        optimizer.zero_grad()
        loss, _, _ = compute_encoder_decoder_loss(
            model, image, texts, tokenizer, max_length=256, label_smoothing=0.1
        )
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass (SAM)
        loss2, _, _ = compute_encoder_decoder_loss(
            model, image, texts, tokenizer, max_length=256, label_smoothing=0.1
        )
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        
        # Update EMA
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()

        # Log training progress
        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter
            logger.info(f'Iter: {nb_iter} | LR: {current_lr:.6f} | Loss: {train_loss_avg:.4f}')
            
            writer.add_scalar('Train/lr', current_lr, nb_iter)
            writer.add_scalar('Train/loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        # Validation
        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = validation_encoder_decoder(
                    model_ema.ema, None, val_loader, tokenizer, device='cuda', max_length=256
                )

                logger.info(f'Validation - Loss: {val_loss:.4f} | CER: {val_cer:.4f} | WER: {val_wer:.4f}')
                writer.add_scalar('Val/loss', val_loss, nb_iter)
                writer.add_scalar('Val/cer', val_cer, nb_iter)
                writer.add_scalar('Val/wer', val_wer, nb_iter)

                # Save best models
                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'tokenizer_info': vocab_info,
                        'args': vars(args)
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'tokenizer_info': vocab_info,
                        'args': vars(args)
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                # Show some predictions
                logger.info('Sample predictions:')
                for i in range(min(3, len(preds))):
                    logger.info(f'  GT: {labels[i]}')
                    logger.info(f'  Pred: {preds[i]}')

            model.train()

    logger.info('Training completed!')
    logger.info(f'Best CER: {best_cer:.4f}')
    logger.info(f'Best WER: {best_wer:.4f}')


if __name__ == '__main__':
    main()
