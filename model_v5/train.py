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
from utils import vn_tags
from functools import partial
import random
import numpy as np
import re
import importlib


def _ctc_compute_gamma_logspace(logp_t_c, y_u):
    """
    Compute CTC posteriors gamma_{t,u} in log-space for a single sample.

    Args:
        logp_t_c: Tensor [T, C] of log probabilities (log_softmax) over base classes, C includes blank at index 0.
        y_u: LongTensor [U] of target class indices in 1..C-1 (no blanks).

    Returns:
        weights: Tensor [U, T] where each row u is normalized over t to sum to 1 (or 0 if degenerate).
                 Detached (no grad) by default for stability.
    """
    device = logp_t_c.device
    T, C = logp_t_c.shape
    U = int(y_u.numel())
    if U == 0 or T == 0:
        return torch.zeros((0, T), device=device)

    # Build extended target sequence l' with blanks inserted: [blank, y1, blank, y2, ..., blank]
    S = 2 * U + 1
    blank = 0
    lprime = torch.full((S,), blank, dtype=torch.long, device=device)
    lprime[1::2] = y_u  # odd positions are labels

    # alpha and beta in log-space
    neg_inf = -1e9
    alpha = torch.full((T, S), neg_inf, device=device)
    beta = torch.full((T, S), neg_inf, device=device)

    # Initialization (t=0)
    alpha[0, 0] = logp_t_c[0, lprime[0]]
    if S > 1:
        alpha[0, 1] = logp_t_c[0, lprime[1]]

    # Forward recursion
    for t in range(1, T):
        lp = logp_t_c[t]
        # vectorized over s
        stay = alpha[t - 1]
        shift1 = torch.full((S,), neg_inf, device=device)
        shift1[1:] = alpha[t - 1, :-1]
        shift2 = torch.full((S,), neg_inf, device=device)
        # allowed skip from s-2 if current label not blank and not equal to l'[s-2]
        mask_skip = torch.zeros((S,), dtype=torch.bool, device=device)
        mask_skip[2:] = (lprime[2:] != blank) & (lprime[2:] != lprime[:-2])
        shift2[mask_skip] = alpha[t - 1, :-2][mask_skip[2:]]
        pre = torch.logsumexp(torch.stack((stay, shift1, shift2), dim=0), dim=0)
        alpha[t] = pre + lp[lprime]

    # Termination probability logZ
    if S == 1:
        logZ = alpha[T - 1, 0]
    else:
        logZ = torch.logsumexp(torch.stack((alpha[T - 1, S - 1], alpha[T - 1, S - 2])), dim=0)

    # Backward initialization at t=T-1
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0

    # Backward recursion
    for t in range(T - 2, -1, -1):
        lp_next = logp_t_c[t + 1]
        # stay
        stay = beta[t + 1] + lp_next[lprime]
        # move to s+1
        shift1 = torch.full((S,), neg_inf, device=device)
        shift1[:-1] = beta[t + 1, 1:] + lp_next[lprime[1:]]
        # skip to s+2 if allowed
        shift2 = torch.full((S,), neg_inf, device=device)
        mask_skip_f = torch.zeros((S,), dtype=torch.bool, device=device)
        mask_skip_f[:-2] = (lprime[:-2] != blank) & (lprime[:-2] != lprime[2:])
        shift2[mask_skip_f] = beta[t + 1, 2:][mask_skip_f[:-2]] + lp_next[lprime[2:]][mask_skip_f[:-2]]
        beta[t] = torch.logsumexp(torch.stack((stay, shift1, shift2), dim=0), dim=0)

    # Gammas for label positions (odd s) -> map to u index
    s_idx = torch.arange(1, S, 2, device=device)  # odd positions
    # gamma_log: [T, U]
    gamma_log = alpha[:, s_idx] + beta[:, s_idx] - logZ
    # Convert to weights over t per u (normalize over time)
    # Avoid underflow by subtract max per column before exp
    m = gamma_log.max(dim=0).values  # [U]
    gamma = torch.exp(gamma_log - m)  # [T, U]
    gamma = gamma.clamp_min(0)
    w = gamma / (gamma.sum(dim=0, keepdim=True) + 1e-8)  # [T, U]
    # Return [U, T]
    return w.transpose(0, 1).detach()


def compute_loss(args, model, image, batch_size, criterion, enc):
    """Compute loss for base head and optionally tag heads when dual-head is enabled.
    enc: output of converter.encode(labels). For CTC converter it's (text, length),
         for DualLabelConverter it's (text_base, length_base, tags_mod, tags_tone, per_sample_U)
    """
    outputs = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    use_dual = isinstance(outputs, dict)

    # Single-head CTC path
    if not use_dual:
        text, length = enc
        preds = outputs.float()
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(preds.device)
        preds = preds.permute(1, 0, 2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        loss = criterion(preds, text.to(preds.device), preds_size, length.to(preds.device)).mean()
        torch.backends.cudnn.enabled = True
        return loss

    # Dual-head: base CTC + tag losses with gamma pooling
    text_base, length_base, tags_mod_list, tags_tone_list, _ = enc
    base = outputs['base'].float()          # [B, T, Cb]
    mod_logits = outputs['mod'].float()     # [B, T, 4]
    tone_logits = outputs['tone'].float()   # [B, T, 6]

    T = int(base.size(1))
    preds_size = torch.IntTensor([T] * batch_size).to(base.device)

    # Base CTC loss
    base_logp = base.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    base_loss = criterion(base_logp, text_base.to(base.device), preds_size, length_base.to(base.device)).mean()
    torch.backends.cudnn.enabled = True

    # Tag losses via gamma pooling
    eps = 1e-8
    batch_mod_losses = []
    batch_tone_losses = []

    # Offsets for slicing flattened targets per sample
    lengths_cpu = [int(x) for x in length_base.detach().cpu().tolist()]
    offs = [0]
    for l in lengths_cpu:
        offs.append(offs[-1] + l)

    # Frame-wise probabilities for tag heads
    mod_sm = torch.softmax(mod_logits, dim=2)   # [B, T, 4]
    tone_sm = torch.softmax(tone_logits, dim=2) # [B, T, 6]

    # Vowel mask lookup for base class ids (0 is blank)
    base_charset_str = utils.build_base_charset()
    vowel_lookup = torch.tensor(
        [0] + [1 if vn_tags.is_vowel(ch) else 0 for ch in base_charset_str],
        device=base.device,
        dtype=torch.float32,
    )

    # Log-probs for base to run CTC forward–backward per sample
    logp_base = torch.log_softmax(base, dim=2)  # [B, T, Cb]

    for b in range(batch_size):
        # Targets for this sample
        y = text_base[offs[b]:offs[b + 1]].to(base.device).long()  # [U]
        U = int(y.numel())
        if U == 0:
            batch_mod_losses.append(torch.tensor(0.0, device=base.device))
            batch_tone_losses.append(torch.tensor(0.0, device=base.device))
            continue

        # Gamma weights over time for each label position: [U, T]
        with torch.no_grad():
            w_ut = _ctc_compute_gamma_logspace(logp_base[b], y)  # [U, T]
        # Renormalize per label to avoid tiny drift
        w_sum = w_ut.sum(dim=1, keepdim=True)
        w_ut = torch.where(w_sum > 0, w_ut / (w_sum + eps), w_ut)

        # Gamma-weighted pooling of tag probabilities
        pooled_mod = w_ut @ mod_sm[b]    # [U, 4]
        pooled_tone = w_ut @ tone_sm[b]  # [U, 6]

        # CE on pooled distributions
        y_mod = tags_mod_list[b].to(base.device).long()   # [U]
        y_tone = tags_tone_list[b].to(base.device).long() # [U]
        ce_mod_u = -torch.log(torch.gather(pooled_mod.clamp_min(1e-8), 1, y_mod.view(-1, 1)).squeeze(1))
        ce_tone_u = -torch.log(torch.gather(pooled_tone.clamp_min(1e-8), 1, y_tone.view(-1, 1)).squeeze(1))

        # Weighted average: vowels=1, consonants=alpha
        vowel_mask = vowel_lookup[y]  # {0,1}
        alpha = float(getattr(args, 'consonant_weight', 0.2))
        weights_u = vowel_mask + alpha * (1.0 - vowel_mask)
        denom = weights_u.sum()
        if float(denom.item()) > 0:
            L_mod = (weights_u * ce_mod_u).sum() / (denom + eps)
            L_tone = (weights_u * ce_tone_u).sum() / (denom + eps)
        else:
            L_mod = torch.tensor(0.0, device=base.device)
            L_tone = torch.tensor(0.0, device=base.device)
        batch_mod_losses.append(L_mod)
        batch_tone_losses.append(L_tone)

    mod_loss = torch.stack(batch_mod_losses).mean()
    tone_loss = torch.stack(batch_tone_losses).mean()

    return base_loss + args.lambda_mod * mod_loss + args.lambda_tone * tone_loss


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

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
            logger.warning(
                f"Failed to initialize wandb: {e}. Continuing without wandb.")
            wandb = None
    else:
        wandb = None

    if args.use_dual_head:
        base_charset_str = utils.build_base_charset()
        nb_cls = len(base_charset_str) + 1
    else:
        full_charset_str = utils.build_full_charset()
        nb_cls = len(full_charset_str) + 1
    model = HTR_VT.create_model(
        nb_cls=nb_cls, img_size=args.img_size[::-1], use_dual_head=args.use_dual_head)

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
    resume_path = args.resume if getattr(
        args, 'resume', None) else getattr(args, 'resume_checkpoint', None)
    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = utils.load_checkpoint(
        model, model_ema, None, resume_path, logger)

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size, ralph=utils.VIETNAMESE_CHARACTERS)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
        prefetch_factor=2,
        collate_fn=partial(dataset.SameTrCollate, args=args),
    )
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=utils.VIETNAMESE_CHARACTERS)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
        prefetch_factor=2,
    )

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    if args.use_dual_head:
        base_charset_str = utils.build_base_charset()
        converter = utils.DualLabelConverter(base_charset_str)
    else:
        full_charset_str = utils.build_full_charset()
        converter = utils.CTCLabelConverter(full_charset_str)

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

        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda(non_blocking=True)
        enc = converter.encode(batch[1])
        batch_size = image.size(0)
        loss = compute_loss(args, model, image, batch_size, criterion, enc)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(args, model, image, batch_size, criterion, enc).backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            logger.info(
                f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ')

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
                    logger.info(
                        f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
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
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(
                        f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
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
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_WER.pth'))

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
