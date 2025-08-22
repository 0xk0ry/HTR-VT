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
from utils import vn_tags
# AMP removed per user request


def compute_loss(args, model, image, batch_size, criterion, enc):
    """Compute loss for base head and optionally tag heads when dual-head is enabled.
    enc: output of converter.encode(labels). For CTC converter it's (text, length),
         for DualLabelConverter it's (text_base, length_base, tags_mod, tags_tone, per_sample_U)
    """
    outputs = model(image, args.mask_ratio,
                    args.max_span_length, use_masking=True)
    use_dual = isinstance(outputs, dict)

    if not use_dual:
        text, length = enc
        preds = outputs.float()
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(preds.device)
        preds = preds.permute(1, 0, 2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        loss = criterion(preds, text.to(preds.device), preds_size, length.to(preds.device)).mean()
        torch.backends.cudnn.enabled = True
        return loss

    # Dual-head path
    text_base, length_base, tags_mod_list, tags_tone_list, _ = enc
    base = outputs['base'].float()  # [B, T, Cb]
    mod_logits = outputs['mod'].float()  # [B, T, 4]
    tone_logits = outputs['tone'].float()  # [B, T, 6]
    T = base.size(1)
    preds_size = torch.IntTensor([T] * batch_size).to(base.device)


    # Base CTC (unchanged)
    base_logp = base.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    base_loss = criterion(base_logp, text_base.to(base.device),
                        preds_size, length_base.to(base.device)).mean()
    torch.backends.cudnn.enabled = True

    # CTC posterior (gamma) pooling for tag supervision
    eps = 1e-8
    batch_mod_losses = []
    batch_tone_losses = []

    # Cache base charset + vowel mask (per run) to avoid rebuilding each call
    if not hasattr(args, '_base_is_vowel_cache'):
        base_charset_str = utils.build_base_charset()
        args._base_charset_len = len(base_charset_str)
        args._base_is_vowel_cache = torch.tensor(
            [vn_tags.is_vowel(ch) for ch in base_charset_str], dtype=torch.bool, device=base.device)
    Cb = args._base_charset_len
    base_is_vowel = args._base_is_vowel_cache  # [Cb]
    tag_alpha = getattr(args, 'tag_alpha', 0.2)

    # Prepare cumulative offsets to slice text_base flat
    lengths_cpu = length_base.detach().cpu().tolist()
    flat_indices = text_base.detach().cpu().tolist()
    offs = [0]
    for l in lengths_cpu:
        offs.append(offs[-1] + l)

    # Precompute log-softmax for tag heads; we'll use probs via exp()
    logp_mod = torch.nn.functional.log_softmax(mod_logits, dim=2)  # [B, T, 4]
    logp_tone = torch.nn.functional.log_softmax(tone_logits, dim=2)  # [B, T, 6]

    def logsumexp_tensor(vals):
        return torch.logsumexp(vals, dim=0)

    for b in range(batch_size):
        # Ground-truth base indices for this sample (1..Cb); 0 is blank
        y = flat_indices[offs[b]:offs[b+1]]  # length U
        U = len(y)
        if U == 0:
            # No characters => no tag loss for this sample
            batch_mod_losses.append(torch.tensor(0.0, device=base.device))
            batch_tone_losses.append(torch.tensor(0.0, device=base.device))
            continue

        y_mod = tags_mod_list[b].detach().cpu().tolist()  # [U]
        y_tone = tags_tone_list[b].detach().cpu().tolist()  # [U]

        # Vowel mask using cached boolean tensor (blank=0 handled)
        y_tensor = torch.tensor(y, device=base.device, dtype=torch.long)
        valid = (y_tensor > 0) & (y_tensor <= Cb)
        vowel_mask = torch.zeros(U, device=base.device, dtype=torch.float32)
        if valid.any():
            vowel_mask[valid] = base_is_vowel[(y_tensor[valid] - 1)].float()

        # Base log-probs time-major for CTC DP: [T, C]
        logp_base_bt = base_logp[:, b, :]  # [T, C]
        T_b = logp_base_bt.size(0)
        C = logp_base_bt.size(1)

        # Build extended target l' with blanks between and at ends
        S = 2 * U + 1
        lprime = torch.zeros(S, dtype=torch.long, device=base.device)
        for u in range(U):
            lprime[2 * u + 1] = int(y[u])

        neg_inf = torch.tensor(-1e30, device=base.device)

        # Forward DP (alpha): [T, S] vectorized across states
        alpha = torch.full((T_b, S), neg_inf.item(), device=base.device)
        # t=0 initialization
        alpha0 = alpha[0]
        alpha0[0] = logp_base_bt[0, 0]
        if S > 1:
            alpha0[1] = logp_base_bt[0, lprime[1]]
        # Precompute label inequality mask for skip transitions (s -> s+2)
        if S > 2:
            skip_ok = (lprime[2:] != lprime[:-2])  # shape [S-2]
        for t in range(1, T_b):
            prev = alpha[t - 1]  # [S]
            same = prev  # stay
            shift1 = torch.cat([neg_inf.view(1), prev[:-1]])  # advance by 1
            if S > 2:
                shift2_core = prev[:-2] + torch.where(skip_ok, torch.zeros_like(prev[:-2]), neg_inf)
                shift2 = torch.cat([neg_inf.view(1), neg_inf.view(1), shift2_core])
            else:
                shift2 = torch.full((S,), neg_inf.item(), device=base.device)
            stacked = torch.stack((same, shift1, shift2), dim=0)  # [3, S]
            alpha[t] = torch.logsumexp(stacked, dim=0) + logp_base_bt[t, lprime]

        # logZ from final states (last blank or last label)
        last_blank = alpha[T_b - 1, S - 1]
        last_label = alpha[T_b - 1, S - 2] if S - 2 >= 0 else neg_inf
        logZ = logsumexp_tensor(torch.stack((last_blank, last_label)))

        # Backward DP (beta): [T, S] vectorized
        beta = torch.full((T_b, S), neg_inf.item(), device=base.device)
        beta_last = beta[T_b - 1]
        beta_last[S - 1] = 0.0
        if S - 2 >= 0:
            beta_last[S - 2] = 0.0
        if S > 2:
            skip_ok_fwd = skip_ok  # reuse
        for t in range(T_b - 2, -1, -1):
            nxt = beta[t + 1]  # [S]
            # stay
            b0 = nxt + logp_base_bt[t + 1, lprime]
            # advance by 1
            b1 = torch.full((S,), neg_inf.item(), device=base.device)
            b1[:-1] = nxt[1:] + logp_base_bt[t + 1, lprime[1:]]
            # advance by 2 where labels differ
            if S > 2:
                b2 = torch.full((S,), neg_inf.item(), device=base.device)
                adv2_src = nxt[2:] + logp_base_bt[t + 1, lprime[2:]]
                # apply mask
                adv2_src = torch.where(skip_ok_fwd, adv2_src, torch.full_like(adv2_src, neg_inf.item()))
                b2[:-2] = adv2_src
                stacked_b = torch.stack((b0, b1, b2), dim=0)
            else:
                stacked_b = torch.stack((b0, b1), dim=0)
            beta[t] = torch.logsumexp(stacked_b, dim=0)

        # Posterior gamma over (t, s)
        with torch.no_grad():
            gamma_log = alpha + beta - logZ
            gamma = torch.exp(gamma_log).clamp_min(0.0)  # [T, S]

        # Per-character CE placeholders
        ce_mod_u = torch.zeros(U, device=base.device)
        ce_tone_u = torch.zeros(U, device=base.device)
        have_pred = torch.zeros(U, device=base.device)

        # Tag probabilities per frame
        p_mod_bt = torch.exp(logp_mod[b])  # [T, 4]
        p_tone_bt = torch.exp(logp_tone[b])  # [T, 6]

        for u_idx in range(U):
            s_idx = 2 * u_idx + 1
            if s_idx >= S:
                continue
            weights_t = gamma[:, s_idx]  # [T]
            w_sum = weights_t.sum()
            if w_sum <= eps:
                continue
            w_norm = (weights_t / w_sum).detach()  # stop-grad through gamma

            # Expected tag distributions for this character
            p_mod_u = (w_norm.unsqueeze(-1) * p_mod_bt).sum(dim=0)  # [4]
            p_tone_u = (w_norm.unsqueeze(-1) * p_tone_bt).sum(dim=0)  # [6]

            ce_val_mod = -torch.log(p_mod_u[y_mod[u_idx]].clamp_min(1e-8))
            ce_val_tone = -torch.log(p_tone_u[y_tone[u_idx]].clamp_min(1e-8))
            ce_mod_u[u_idx] = ce_val_mod
            ce_tone_u[u_idx] = ce_val_tone
            have_pred[u_idx] = 1.0

        # Masked average over vowel positions where we had predictions
        weights_chars = vowel_mask + tag_alpha * (1.0 - vowel_mask)
        eff_w = weights_chars * have_pred
        denom = eff_w.sum()
        L_mod = (eff_w * ce_mod_u).sum() / (denom + 1e-6) if denom > 0 else torch.tensor(0.0, device=base.device)
        L_tone = (eff_w * ce_tone_u).sum() / (denom + 1e-6) if denom > 0 else torch.tensor(0.0, device=base.device)
        batch_mod_losses.append(L_mod)
        batch_tone_losses.append(L_tone)

    # Average across batch
    mod_loss = torch.stack(batch_mod_losses).mean() if batch_mod_losses else torch.tensor(0.0, device=base.device)
    tone_loss = torch.stack(batch_tone_losses).mean() if batch_tone_losses else torch.tensor(0.0, device=base.device)

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
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=args.img_size[::-1], use_dual_head=args.use_dual_head)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    # Performance toggles
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')  # enable TF32 if supported
    except Exception:
        pass

    model.train()
    model = model.cuda().to(memory_format=torch.channels_last)
    # Ensure EMA decay is properly accessed (handle both ema_decay and ema-decay)
    ema_decay = getattr(args, 'ema_decay', 0.9999)
    logger.info(f"Using EMA decay: {ema_decay}")
    model_ema = utils.ModelEma(model, ema_decay)
    model.zero_grad()

    # AMP disabled per user request

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

    # Use fused AdamW when available (PyTorch 2.0+ with CUDA)
    fused_ok = torch.cuda.is_available() and hasattr(torch.optim, 'AdamW') and \
        'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    base_opt = partial(torch.optim.AdamW, fused=True) if fused_ok else torch.optim.AdamW
    optimizer = sam.SAM(model.parameters(), base_opt,
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

        # Update LR
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer
        )

        # Get batch
        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda(non_blocking=True).to(memory_format=torch.channels_last)
        enc = converter.encode(batch[1])
        batch_size = image.size(0)

        # First forward/backward (fp32)
        loss = compute_loss(args, model, image, batch_size, criterion, enc)
        loss.backward()

        optimizer.first_step(zero_grad=True)

        # Second forward/backward (fp32)
        loss2 = compute_loss(args, model, image, batch_size, criterion, enc)
        loss2.backward()
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
