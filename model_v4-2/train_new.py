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
import torch.nn.functional as F
import wandb


def compute_loss(
    args,
    model,
    image,
    batch_size,
    criterion,
    text,
    length,
    labels,
    converter,
    tau_v,
    lambda_tone,
    tone_loss_type,
    focal_gamma,
    lang_weight,
    global_step,
    use_masking: bool = True,
):
    # Forward model: get base and tone logits
    outputs = model(image, args.mask_ratio,
                    args.max_span_length, use_masking=use_masking)
    if isinstance(outputs, (list, tuple)):
        base_logits, tone_logits = outputs
    else:
        # Backward compatibility: older checkpoints without tone head
        base_logits, tone_logits = outputs, None
    base_logits = base_logits.float()

    # CTC loss on base logits
    preds_size = torch.IntTensor([base_logits.size(1)] * batch_size).cuda()
    log_probs_tbc = base_logits.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    ctc_loss = criterion(log_probs_tbc, text.cuda(),
                         preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True

    logs = {
        'ctc_loss': ctc_loss.detach().item(),
        'tone_loss': 0.0,
        'gate_hit_rate': 0.0,
        'tone_leakage': 0.0,
        'lambda_tone_eff': 0.0,
    }

    # Curriculum: decide effective lambda for this step
    if global_step < args.tone_warmup_iters:
        lambda_eff = 0.0
    elif global_step < args.tone_warmup_iters + args.tone_ramp_iters:
        ratio = (global_step - args.tone_warmup_iters) / \
            max(1, args.tone_ramp_iters)
        lambda_eff = lambda_tone * float(ratio)
    else:
        lambda_eff = lambda_tone
    logs['lambda_tone_eff'] = float(lambda_eff)

    tone_loss_total = torch.tensor(0.0, device=base_logits.device)
    gate_hits = []
    leakage_vals = []

    if tone_logits is not None and lambda_eff > 0.0:
        # Get base probabilities for gating
        base_probs = F.softmax(base_logits, dim=-1)  # (B, T, C)
        # Vowel indices from converter
        vowel_idxs = utils.vowel_indices_from_converter(converter)
        if len(vowel_idxs) > 0:
            vowel_mask = torch.zeros(
                base_probs.size(-1), device=base_probs.device)
            vowel_mask.scatter_(0, torch.tensor(
                vowel_idxs, device=base_probs.device, dtype=torch.long), 1.0)
            # Gate per-frame
            v_score = (base_probs * vowel_mask.view(1, 1, -1)
                       ).sum(dim=-1)  # (B, T)
            m_t = (v_score >= tau_v).float()  # (B, T)
        else:
            m_t = torch.ones(base_probs.shape[:2], device=base_probs.device)

        # Tone probabilities
        tone_log_probs = F.log_softmax(tone_logits, dim=-1)  # (B, T, 6)
        tone_probs = tone_log_probs.exp()

        # Build gamma using batched CTC posteriors
        log_probs_base = F.log_softmax(base_logits, dim=-1).detach()
        # Prepare targets for batched CTC posteriors
        targets = []
        cursor = 0
        for b in range(batch_size):
            U_b = int(length[b].item())
            y_b = text[cursor:cursor + U_b]
            targets.append(y_b)
            cursor += U_b
        gammas = utils.ctc_posteriors_batched(log_probs_base, targets, blank=0)

        # Iterate over batch
        tone_losses = []
        for b in range(batch_size):
            U_b = int(length[b].item())
            gamma_b = gammas[b].detach()  # (T, U_b)
            label_b = labels[b]
            # Derive tone id per original character (length U_b)
            tones_j = [utils.tone_of_char(ch) for ch in label_b]
            if len(tones_j) != U_b:
                # Fallback: pad or truncate to match lengths
                tones_j = (tones_j + [0] * U_b)[:U_b]

            # Soft target over tones per frame: sum_j gamma[t,j] * one_hot(tone_j)
            T_b = gamma_b.size(0)
            target_t = torch.zeros(T_b, 6, device=base_logits.device)
            if U_b > 0:
                one_hot = torch.zeros(U_b, 6, device=base_logits.device)
                one_hot[torch.arange(U_b, device=base_logits.device), torch.tensor(
                    tones_j, device=base_logits.device)] = 1.0
                target_t = gamma_b @ one_hot  # (T, 6)

            # Apply language weight
            lang_w = lang_weight if utils.is_english_label(label_b) else 1.0

            # Framewise loss
            if tone_loss_type == 'focal':
                pt = tone_probs[b]  # (T, 6)
                loss_b = - (target_t * ((1 - pt) ** focal_gamma)
                            * tone_log_probs[b]).sum(dim=-1)
            else:
                # Soft CE
                loss_b = - (target_t * tone_log_probs[b]).sum(dim=-1)

            # Gate and average
            gate_b = m_t[b]  # (T,)
            denom = gate_b.sum() + 1e-6
            loss_b = (gate_b * loss_b).sum() / denom
            tone_losses.append(loss_b * lang_w)

            # Metrics
            gate_hits.append(gate_b.mean().detach())
            # Leakage: encourage NONE when gate is 0 -> measure non-NONE prob under gate=0
            none_prob = tone_probs[b, :, 0]
            leak = (1 - none_prob) * (1 - gate_b)
            denom_leak = (1 - gate_b).sum() + 1e-6
            leakage_vals.append(leak.sum() / denom_leak)

        if tone_losses:
            tone_loss_total = torch.stack(tone_losses).mean()
            logs['tone_loss'] = tone_loss_total.detach().item()
            logs['gate_hit_rate'] = torch.stack(gate_hits).mean().item()
            logs['tone_leakage'] = torch.stack(leakage_vals).mean().item()

    total_loss = ctc_loss + lambda_eff * tone_loss_total
    return total_loss, logs


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
            import wandb  # type: ignore
            wandb.init(project="HTR-VN", name=args.exp_name,
                       config=vars(args), dir=args.save_dir)
        except Exception as e:
            print(f"W&B initialization failed, continuing without it: {e}")
            args.use_wandb = False

    model = HTR_VT.create_model(
        nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    # Ensure EMA decay is properly accessed (handle both ema_decay and ema-decay)
    ema_decay = getattr(args, 'ema_decay', 0.9999)
    logger.info(f"Using EMA decay: {ema_decay}")
    model_ema = utils.ModelEma(model, ema_decay)
    model.zero_grad()

    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = utils.load_checkpoint(
        model, model_ema, None, getattr(args, 'resume_checkpoint', None), logger)

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    logger.info('Initializing optimizer, criterion and converter...')
    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    # Load optimizer state after initialization
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            logger.info("Successfully loaded optimizer state")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            logger.info(
                "Continuing training without optimizer state (will restart from initial lr/momentum)")

    # --- Helper for overlaying text on image ---
    #### ---- train & eval ---- ####
    logger.info('Start training...')
    # Accumulators for logging
    tone_loss_running = 0.0
    gate_hit_running = 0.0
    leakage_running = 0.0
    tone_count = 0

    for nb_iter in range(start_iter, args.total_iter):
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)
        loss, loss_logs = compute_loss(
            args,
            model,
            image,
            batch_size,
            criterion,
            text,
            length,
            batch[1],  # labels
            converter,
            args.tau_v,
            args.lambda_tone,
            args.tone_loss,
            args.focal_gamma,
            args.lang_weight,
            nb_iter,
            use_masking=True,
        )
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(
            args,
            model,
            image,
            batch_size,
            criterion,
            text,
            length,
            batch[1],
            converter,
            args.tau_v,
            args.lambda_tone,
            args.tone_loss,
            args.focal_gamma,
            args.lang_weight,
            nb_iter,
            use_masking=False,
        )[0].backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)

        train_loss += loss.item()
        train_loss_count += 1
        # Accumulate tone-related logs
        tone_loss_running += loss_logs.get('tone_loss', 0.0)
        gate_hit_running += loss_logs.get('gate_hit_rate', 0.0)
        leakage_running += loss_logs.get('tone_leakage', 0.0)
        tone_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            logger.info(
                f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ')

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            if tone_count > 0:
                writer.add_scalar('./Train/tone_loss',
                                  tone_loss_running / tone_count, nb_iter)
                writer.add_scalar('./Train/gate_hit_rate',
                                  gate_hit_running / tone_count, nb_iter)
                writer.add_scalar('./Train/tone_leakage',
                                  leakage_running / tone_count, nb_iter)

            # wandb log
            log_dict = {"iter": nb_iter, "train/lr": current_lr,
                        "train/loss": train_loss_avg}

            if tone_count > 0:
                log_dict.update({
                    "train/tone_loss": tone_loss_running / tone_count,
                    "train/gate_hit_rate": gate_hit_running / tone_count,
                    "train/tone_leakage": leakage_running / tone_count,
                })

            train_loss = 0.0
            train_loss_count = 0
            tone_loss_running = 0.0
            gate_hit_running = 0.0
            leakage_running = 0.0
            tone_count = 0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter,
                                                                             args)

                # Compute Tone Error Rate (TER) on Vietnamese subset
                val_ter = 0.0
                try:
                    vi_pairs = [(p, g) for p, g in zip(
                        preds, labels) if utils.contains_vietnamese(g)]
                    if len(vi_pairs) > 0:
                        tone_errors = 0
                        tone_total = 0
                        for p, g in vi_pairs:
                            L = min(len(p), len(g))
                            for i in range(L):
                                if utils.is_vietnamese_vowel(g[i]):
                                    tone_total += 1
                                    tone_errors += int(utils.tone_of_char(
                                        p[i]) != utils.tone_of_char(g[i]))
                        val_ter = tone_errors / max(1, tone_total)
                except Exception:
                    val_ter = 0.0

                if val_cer < best_cer:
                    logger.info(
                        f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                if val_wer < best_wer:
                    logger.info(
                        f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer

                if val_cer < best_cer:
                    checkpoint = utils.make_checkpoint_dict(
                        model, model_ema, optimizer, nb_iter, best_cer, best_wer,
                        val_cer, val_wer, val_ter, args, train_loss, train_loss_count
                    )
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    checkpoint = utils.make_checkpoint_dict(
                        model, model_ema, optimizer, nb_iter, best_cer, best_wer,
                        val_cer, val_wer, val_ter, args, train_loss, train_loss_count
                    )
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t TER : {val_ter:0.4f}')

                checkpoint_regular = utils.make_checkpoint_dict(
                    model, model_ema, optimizer, nb_iter, best_cer, best_wer,
                    val_cer, val_wer, val_ter, args, train_loss, train_loss_count
                )
                ckpt_name = f"checkpoint_{nb_iter}_{val_cer:.4f}_{val_wer:.4f}_{val_ter:.4f}.pth"
                torch.save(checkpoint_regular, os.path.join(
                    args.save_dir, ckpt_name))
                logger.info(f'Saved checkpoint: {ckpt_name}')

                writer.add_scalar('./val/CER', val_cer, nb_iter)
                writer.add_scalar('./val/WER', val_wer, nb_iter)
                writer.add_scalar('./val/TER', val_ter, nb_iter)
                writer.add_scalar('./val/bestCER', best_cer, nb_iter)
                writer.add_scalar('./val/bestWER', best_wer, nb_iter)
                writer.add_scalar('./val/val_loss', val_loss, nb_iter)

                if args.use_wandb:
                    example_count = min(5, batch[0].size(0))
                    examples_table = wandb.Table(
                        columns=["image", "ground_truth", "prediction"])

                    model.eval()
                    with torch.no_grad():
                        image = batch[0].cuda()

                        preds = model(image)
                        if isinstance(preds, (list, tuple)):
                            preds = preds[0]
                        preds = preds.float()
                        batch_size = image.size(0)
                        preds_size = torch.IntTensor(
                            [preds.size(1)] * batch_size)
                        preds = preds.permute(1, 0, 2).log_softmax(2)
                        _, preds_index = preds.max(2)
                        preds_index = preds_index.transpose(
                            1, 0).contiguous().view(-1)
                        preds_str = converter.decode(
                            preds_index.data, preds_size.data)

                    for i in range(example_count):
                        img_tensor = batch[0][i].cpu()
                        pred_text = preds_str[i]
                        true_text = batch[1][i]
                        examples_table.add_data(wandb.Image(
                            img_tensor), true_text, pred_text)

                    log_dict.update({
                        "val/loss": val_loss,
                        "val/CER": val_cer,
                        "val/WER": val_wer,
                        "val/TER": val_ter,
                        "val/best_CER": best_cer,
                        "val/best_WER": best_wer,
                        "val/examples_table": examples_table,
                    })
                model.train()

        if args.use_wandb and nb_iter % args.print_iter == 0:
            logger.info(f"Logging to wandb: {log_dict}")
            wandb.log(log_dict)


if __name__ == '__main__':
    main()
