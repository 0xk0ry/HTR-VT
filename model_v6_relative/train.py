import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial
import random
import numpy as np
import re
import importlib


def compute_loss(args, model, image, batch_size, criterion, enc):
    """Compute loss for base head and optionally tone head when tone head is enabled.
    enc: output of converter.encode(labels). For CTC converter it's (text, length),
         for ToneLabelConverter it's (text_base, length_base, tags_tone, per_sample_U)
    """
    outputs = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    use_tone = isinstance(outputs, dict)

    if not use_tone:
        # Single head path (original CTC)
        text, length = enc
        preds = outputs.float()
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
        preds = preds.permute(1, 0, 2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
        torch.backends.cudnn.enabled = True
        return loss

    # Tone head path
    text_base, length_base, tags_tone_list, _ = enc
    base = outputs['base'].float()  # [B, T, Cb]
    tone_logits = outputs['tone'].float()  # [B, T, 6]
    T = base.size(1)
    device = base.device
    preds_size = torch.IntTensor([T] * batch_size).to(device)

    # Base CTC loss
    base_logp = base.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    base_loss = criterion(base_logp, text_base.to(device), preds_size, length_base.to(device)).mean()
    torch.backends.cudnn.enabled = True

    # Check if tone loss should be computed
    lambda_tone = getattr(args, 'lambda_tone', 0.7)
    if lambda_tone == 0.0:
        return base_loss

    # Tone loss computation (adapted from model_v5_mask)
    tag_alpha = getattr(args, 'tag_alpha', 0.2)
    
    # Pre-compute log probabilities
    logp_tone = torch.nn.functional.log_softmax(tone_logits, dim=2)
    
    # Get base predictions for alignment
    with torch.no_grad():
        base_argmax = base.argmax(dim=2)  # [B, T]

    # Optimized batch processing
    lengths_cpu = length_base.detach().cpu().tolist()
    flat_indices = text_base.detach().cpu().tolist()
    
    # Cache vowel information
    if not hasattr(compute_loss, '_vowel_chars'):
        from utils import vn_tone_tags
        compute_loss._vowel_chars = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY')
        compute_loss._vn_tone_tags = vn_tone_tags
    
    vowel_chars = compute_loss._vowel_chars
    base_charset_str = utils.build_base_charset()
    
    # Process each sample in batch
    batch_tone_losses = torch.zeros(batch_size, device=device)
    batch_valid_mask = torch.zeros(batch_size, device=device)
    
    flat_idx = 0
    for b in range(batch_size):
        U = lengths_cpu[b]
        if U == 0:
            continue
            
        # Get ground truth for this sample
        y = flat_indices[flat_idx:flat_idx + U]
        y_tone = tags_tone_list[b].cpu().tolist()
        flat_idx += U
        
        # Create vowel mask for weighting
        vowel_mask = torch.zeros(U)
        for u in range(U):
            if y[u] > 0 and y[u] <= len(base_charset_str):
                char = base_charset_str[y[u] - 1]
                vowel_mask[u] = 1.0 if char in vowel_chars else 0.0
        
        # Get predicted frames for alignment
        frames = base_argmax[b]  # [T]
        
        # Simple alignment: find segments for each ground truth position
        non_blank_mask = frames != 0
        if not non_blank_mask.any():
            continue
            
        # Extract frame segments
        prev_frames = torch.cat([torch.tensor([0], device=device), frames[:-1]])
        change_points = (frames != prev_frames) & non_blank_mask
        segment_starts = torch.nonzero(change_points, as_tuple=True)[0]
        
        if len(segment_starts) == 0:
            continue
            
        # Map segments to ground truth positions
        segment_values = frames[segment_starts]
        valid_segments = []
        
        for start_idx in range(len(segment_starts)):
            start_pos = segment_starts[start_idx]
            end_pos = segment_starts[start_idx + 1] if start_idx + 1 < len(segment_starts) else len(frames)
            segment_frames = torch.arange(start_pos, end_pos, device=device)
            segment_class = segment_values[start_idx].item()
            valid_segments.append((segment_class, segment_frames))
        
        # Match ground truth to segments
        gt_to_segment = {}
        seg_idx = 0
        for u, gt_class in enumerate(y):
            while seg_idx < len(valid_segments) and valid_segments[seg_idx][0] != gt_class:
                seg_idx += 1
            if seg_idx < len(valid_segments):
                gt_to_segment[u] = valid_segments[seg_idx][1]
                seg_idx += 1
        
        # Compute tone losses
        if gt_to_segment:
            tone_losses = []
            weights = []
            
            for u in range(U):
                if u in gt_to_segment:
                    t_indices = gt_to_segment[u]
                    if len(t_indices) > 0:
                        # Average tone predictions over segment frames
                        lpt = logp_tone[b, t_indices, :].mean(dim=0)
                        tone_loss = (-lpt[y_tone[u]]).clamp_min(-60.0)
                        tone_losses.append(tone_loss)
                        
                        # Weight: 1.0 for vowels, tag_alpha for consonants
                        weight = 1.0 if vowel_mask[u] > 0.5 else tag_alpha
                        weights.append(weight)
            
            if tone_losses:
                # Weighted average
                tone_tensor = torch.stack(tone_losses)
                weight_tensor = torch.tensor(weights, device=device)
                total_weight = weight_tensor.sum() + 1e-6
                batch_tone_losses[b] = (weight_tensor * tone_tensor).sum() / total_weight
                batch_valid_mask[b] = 1.0
    
    # Final loss aggregation
    valid_samples = batch_valid_mask.sum()
    if valid_samples > 0:
        tone_loss = (batch_tone_losses * batch_valid_mask).sum() / valid_samples
    else:
        tone_loss = torch.tensor(0.0, device=device)
    
    return base_loss + lambda_tone * tone_loss


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    # Initialize wandb only if enabled
    if getattr(args, 'use_wandb', False):
        try:
            wandb = importlib.import_module('wandb')
            wandb.init(project=getattr(args, 'wandb_project', 'None'), name=args.exp_name,
                       config=vars(args), dir=args.save_dir)
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb.")
            wandb = None
    else:
        wandb = None

    # Initialize model with tone head support
    if getattr(args, 'use_tone_head', False):
        base_charset_str = utils.build_base_charset()
        nb_cls = len(base_charset_str) + 1
        logger.info(f"Using tone head mode with base charset size: {nb_cls}")
        model = HTR_VT.create_model(nb_cls=nb_cls, img_size=args.img_size[::-1], use_tone_head=True)
    else:
        model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
        logger.info(f"Using single head mode with charset size: {args.nb_cls}")

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    # Ensure EMA decay is properly accessed (handle both ema_decay and ema-decay)
    ema_decay = getattr(args, 'ema_decay', 0.9999)
    logger.info(f"Using EMA decay: {ema_decay}")
    model_ema = utils.ModelEma(model, ema_decay)
    model.zero_grad()

    # Use centralized checkpoint loader like model_v4-2
    resume_path = args.resume
    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = utils.load_checkpoint(
        model, model_ema, None, resume_path, logger)

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    
    # Initialize appropriate converter
    if getattr(args, 'use_tone_head', False):
        converter = utils.ToneLabelConverter(utils.build_base_charset())
        logger.info("Using ToneLabelConverter for tone head training")
    else:
        converter = utils.CTCLabelConverter(train_dataset.ralph.values())
        logger.info("Using CTCLabelConverter for standard training")

    # Load optimizer state after initialization
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            logger.info("Successfully loaded optimizer state")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            logger.info(
                "Continuing training without optimizer state (will restart from initial lr/momentum)")

    best_cer, best_wer = best_cer, best_wer
    train_loss = train_loss
    train_loss_count = train_loss_count
    #### ---- train & eval ---- ####
    logger.info('Start training...')
    for nb_iter in range(start_iter, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()

        if getattr(args, 'use_tone_head', False):
            # Tone head mode: encode returns (text_base, length_base, tags_tone, per_sample_U)
            enc = converter.encode(batch[1])
            text_base, length_base, tags_tone_list, per_sample_U = enc
            batch_size = image.size(0)
            loss = compute_loss(args, model, image, batch_size, criterion, enc)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, image, batch_size, criterion, enc).backward()
        else:
            # Single head mode: encode returns (text, length)
            text, length = converter.encode(batch[1])
            batch_size = image.size(0)
            loss = compute_loss(args, model, image, batch_size, criterion, (text, length))
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, image, batch_size, criterion, (text, length)).backward()

        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            if wandb is not None:
                wandb.log({
                    'train/lr': current_lr,
                    'train/loss': train_loss_avg,
                    'iter': nb_iter,
                }, step=nb_iter)
            train_loss = 0.0
            train_loss_count = 0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)
                # Save checkpoint every print interval (like model_v4-2)
                ckpt_name = f"checkpoint_{best_cer:.4f}_{best_wer:.4f}_{nb_iter}.pth"
                checkpoint = {
                    'model': model.state_dict(),
                    'state_dict_ema': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'nb_iter': nb_iter,
                    'best_cer': best_cer,
                    'best_wer': best_wer,
                    'args': vars(args),
                    'random_state': random.getstate(),
                    'numpy_state': np.random.get_state(),
                    'torch_state': torch.get_rng_state(),
                    'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'train_loss': train_loss,
                    'train_loss_count': train_loss_count,
                }
                torch.save(checkpoint, os.path.join(args.save_dir, ckpt_name))
                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                if wandb is not None:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/CER': val_cer,
                        'val/WER': val_wer,
                        'val/best_CER': best_cer,
                        'val/best_WER': best_wer,
                        'iter': nb_iter,
                    }, step=nb_iter)
                model.train()


if __name__ == '__main__':
    main()