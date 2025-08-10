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
    outputs = model(image, args.mask_ratio, args.max_span_length, use_masking=use_masking)
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
    ctc_loss = criterion(log_probs_tbc, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True

    logs = {
        'ctc_loss': ctc_loss.detach().item(),
        'tone_loss': 0.0,
        'guard_loss': 0.0,
        'gate_hit_rate': 0.0,
        'tone_leakage': 0.0,
        'lambda_tone_eff': 0.0,
        # Diagnostics placeholders
        'blank_rate': 0.0,
        'frames_per_vowel': 0.0,
        'T_over_U_mean': 0.0,
        'T_over_U_p05': 0.0,
        'pct_T_over_U_lt_1p5': 0.0,
        'tone_none_rate_on_vowel': 0.0,
        'tone_conf_margin': 0.0,
        'tone_class_support': {},
        'mask_ratio_effective': float(getattr(args, 'mask_ratio', 0.0)),
        'mask_span_hits': 0.0,
    }

    # Curriculum: decide effective lambda for this step
    if global_step < args.tone_warmup_iters:
        lambda_eff = 0.0
    elif global_step < args.tone_warmup_iters + args.tone_ramp_iters:
        ratio = (global_step - args.tone_warmup_iters) / max(1, args.tone_ramp_iters)
        lambda_eff = lambda_tone * float(ratio)
    else:
        lambda_eff = lambda_tone
    logs['lambda_tone_eff'] = float(lambda_eff)

    tone_loss_total = torch.tensor(0.0, device=base_logits.device)
    guard_loss_total = torch.tensor(0.0, device=base_logits.device)
    gate_hits = []
    leakage_vals = []
    frames_per_vowel_list = []
    none_on_vowel = 0
    total_vowel_spans = 0
    conf_margins = []
    class_support_counts = torch.zeros(6, device=base_logits.device)

    if tone_logits is not None:
        # Get base probabilities for gating
        base_probs = F.softmax(base_logits, dim=-1)  # (B, T, C)
        # Vowel indices from converter
        vowel_idxs = utils.vowel_indices_from_converter(converter)
        if len(vowel_idxs) > 0:
            vowel_mask = torch.zeros(base_probs.size(-1), device=base_probs.device)
            vowel_mask.scatter_(0, torch.tensor(vowel_idxs, device=base_probs.device, dtype=torch.long), 1.0)
            # Gate per-frame
            v_score = (base_probs * vowel_mask.view(1, 1, -1)).sum(dim=-1)  # (B, T)
            m_t = (v_score >= tau_v).float().detach()  # (B, T)
        else:
            m_t = torch.ones(base_probs.shape[:2], device=base_probs.device)

        # Tone probabilities
        tone_log_probs = F.log_softmax(tone_logits, dim=-1)  # (B, T, 6)
        tone_probs = tone_log_probs.exp()

        # Build gamma using CTC posteriors per sample (detached for tone loss)
        log_probs_base = F.log_softmax(base_logits, dim=-1).detach()

        # Iterate over batch
        cursor = 0
        tone_losses = []
        guard_losses = []
        # Compute some batch-level stats independent of labels
        # Blank rate
        frame_labels = base_logits.argmax(dim=-1)  # (B, T)
        blank_rate = (frame_labels == 0).float().mean().item()
        logs['blank_rate'] = blank_rate
        # Masking spans/ratio (approximate from args and sequence length)
        T_len = base_logits.size(1)
        total_mask = int(T_len * getattr(args, 'mask_ratio', 0.0))
        span_hits = max(1, total_mask // max(1, getattr(args, 'max_span_length', 1))) if total_mask > 0 else 0
        logs['mask_span_hits'] = float(span_hits)
        # Alignment T/U stats
        t_over_u_list = []
        for b in range(batch_size):
            U_b = int(length[b].item())
            if U_b > 0:
                t_over_u_list.append(float(T_len) / float(U_b))
            y_b = text[cursor:cursor + U_b]
            cursor += U_b
            # posteriors gamma: (T, U_b)
            gamma_b = utils.ctc_posteriors(log_probs_base[b], y_b, blank=0).detach()
            # Tone targets per label position j
            label_b = labels[b]
            # Derive tone id per original character (length U_b)
            # We need to map back from indices to characters; converter.character holds mapping [blank] + chars.
            # Build token list for the label in the same order
            # The provided labels[b] is the string; we'll derive tones per character directly.
            tones_j = [utils.tone_of_char(ch) for ch in label_b]
            if len(tones_j) != U_b:
                # Fallback: pad or truncate to match lengths
                tones_j = (tones_j + [0] * U_b)[:U_b]

            # Soft target over tones per frame: sum_j gamma[t,j] * one_hot(tone_j)
            T_b = gamma_b.size(0)
            target_t = torch.zeros(T_b, 6, device=base_logits.device)
            if U_b > 0:
                one_hot = torch.zeros(U_b, 6, device=base_logits.device)
                one_hot[torch.arange(U_b, device=base_logits.device), torch.tensor(tones_j, device=base_logits.device)] = 1.0
                target_t = gamma_b @ one_hot  # (T, 6)

            # Apply language weight
            lang_w = lang_weight if utils.is_english_label(label_b) else 1.0

            # Framewise loss
            if lambda_eff > 0.0:
                if tone_loss_type == 'focal':
                    pt = tone_probs[b]  # (T, 6)
                    loss_b = - (target_t * ((1 - pt) ** focal_gamma) * tone_log_probs[b]).sum(dim=-1)
                else:
                    # Soft CE
                    loss_b = - (target_t * tone_log_probs[b]).sum(dim=-1)

                # Gate and average
                gate_b = m_t[b]  # (T,)
                denom = gate_b.sum() + 1e-6
                loss_b = (gate_b * loss_b).sum() / denom
                tone_losses.append(loss_b * lang_w)
            else:
                gate_b = m_t[b]

            # Guard loss: encourage NONE on non-vowel frames
            lambda_guard = float(getattr(args, 'lambda_guard', 0.0))
            if lambda_guard > 0.0:
                none_logp = tone_log_probs[b, :, 0]  # (T,)
                inv_gate = (1 - gate_b)
                denom_g = inv_gate.sum() + 1e-6
                guard_b = - (inv_gate * none_logp).sum() / denom_g
                guard_losses.append(guard_b)

            # Metrics
            gate_hits.append(gate_b.mean().detach())
            # Leakage: encourage NONE when gate is 0 -> measure non-NONE prob under gate=0
            none_prob = tone_probs[b, :, 0]
            leak = (1 - none_prob) * (1 - gate_b)
            denom_leak = (1 - gate_b).sum() + 1e-6
            leakage_vals.append(leak.sum() / denom_leak)

            # Span-based stats using greedy spans over frame_labels
            # Build spans (c_idx, start, end)
            spans = []
            last_c = 0
            start_s = None
            for t in range(T_len):
                c = int(frame_labels[b, t].item())
                if c == 0:
                    if start_s is not None:
                        spans.append((last_c, start_s, t))
                        start_s = None
                    last_c = 0
                    continue
                if start_s is None:
                    start_s = t
                    last_c = c
                elif c != last_c:
                    spans.append((last_c, start_s, t))
                    start_s = t
                    last_c = c
            if start_s is not None:
                spans.append((last_c, start_s, T_len))

            # Iterate spans: only vowel chars
            for (c_idx, s, e) in spans:
                if c_idx <= 0 or c_idx >= len(converter.character):
                    continue
                ch = converter.character[c_idx]
                if not utils.is_vietnamese_vowel(ch):
                    continue
                total_vowel_spans += 1
                # frames used (by gate) in this span
                frames_used = float(gate_b[s:e].sum().item())
                frames_per_vowel_list.append(frames_used)
                # Aggregate tone over span and compute margin
                # Build gate for this span
                g_span = gate_b[s:e]
                probs = F.softmax(tone_logits[b, s:e, :], dim=-1)
                denom_span = g_span.sum()
                if float(denom_span.item()) < 1e-6:
                    g_span = torch.ones_like(g_span)
                    denom_span = g_span.sum()
                avg = (probs * g_span.view(-1, 1)).sum(dim=0) / denom_span
                best_tone = int(avg[1:].argmax().item()) + 1
                margin = float(avg[best_tone] - avg[0])
                kappa = float(getattr(args, 'tone_kappa', 0.3))
                tone_id = best_tone if margin >= kappa else 0
                if tone_id == 0:
                    none_on_vowel += 1
                else:
                    conf_margins.append(margin)
                # class support
                class_support_counts[tone_id] += 1

        if tone_losses:
            tone_loss_total = torch.stack(tone_losses).mean()
            logs['tone_loss'] = tone_loss_total.detach().item()
            logs['gate_hit_rate'] = torch.stack(gate_hits).mean().item()
            logs['tone_leakage'] = torch.stack(leakage_vals).mean().item()
        if guard_losses:
            guard_loss_total = torch.stack(guard_losses).mean()
            logs['guard_loss'] = guard_loss_total.detach().item()

        # Finalize diagnostics
        if len(frames_per_vowel_list) > 0:
            logs['frames_per_vowel'] = float(torch.tensor(frames_per_vowel_list, device=base_logits.device).mean().item())
        if total_vowel_spans > 0:
            logs['tone_none_rate_on_vowel'] = float(none_on_vowel) / float(total_vowel_spans)
            if len(conf_margins) > 0:
                logs['tone_conf_margin'] = float(torch.tensor(conf_margins, device=base_logits.device).mean().item())
            # Normalize class support to fractions
            support_total = float(class_support_counts.sum().item())
            if support_total > 0:
                frac = (class_support_counts / support_total).detach().cpu().tolist()
                logs['tone_class_support'] = {
                    'NONE': frac[0], 'ACUTE': frac[1], 'GRAVE': frac[2], 'HOOK': frac[3], 'TILDE': frac[4], 'DOT': frac[5]
                }
        if len(t_over_u_list) > 0:
            t_over_u_tensor = torch.tensor(t_over_u_list, device=base_logits.device)
            logs['T_over_U_mean'] = float(t_over_u_tensor.mean().item())
            logs['T_over_U_p05'] = float(t_over_u_tensor.kthvalue(max(1, int(0.05 * len(t_over_u_list)))).values.item()) if len(t_over_u_list) > 1 else float(t_over_u_tensor.item())
            logs['pct_T_over_U_lt_1p5'] = float((t_over_u_tensor < 1.5).float().mean().item())

    total_loss = ctc_loss + lambda_eff * tone_loss_total + float(getattr(args, 'lambda_guard', 0.0)) * guard_loss_total
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

    # Helper to load checkpoint (consistent with test.py)
    def load_checkpoint(model, model_ema, optimizer, checkpoint_path):
        from collections import OrderedDict
        import re
        
        best_cer, best_wer, start_iter = 1e+6, 1e+6, 1
        train_loss, train_loss_count = 0.0, 0
        optimizer_state = None
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load model state dict (handle module prefix like in test.py)
            model_dict = OrderedDict()
            pattern = re.compile('module.')
            
            # For main model, load from the 'model' state dict 
            # (the training checkpoint contains both 'model' and 'state_dict_ema')
            if 'model' in checkpoint:
                source_dict = checkpoint['model']
                logger.info("Loading main model from 'model' state dict")
            elif 'state_dict_ema' in checkpoint:
                source_dict = checkpoint['state_dict_ema']
                logger.info("Loading main model from 'state_dict_ema' (fallback)")
            else:
                raise KeyError("Neither 'model' nor 'state_dict_ema' found in checkpoint")
            
            for k, v in source_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict[k] = v
            
            model.load_state_dict(model_dict, strict=True)
            logger.info("Successfully loaded main model state dict")
            
            # Load EMA state dict if available
            if 'state_dict_ema' in checkpoint and model_ema is not None:
                ema_dict = OrderedDict()
                for k, v in checkpoint['state_dict_ema'].items():
                    if re.search("module", k):
                        ema_dict[re.sub(pattern, '', k)] = v
                    else:
                        ema_dict[k] = v
                model_ema.ema.load_state_dict(ema_dict, strict=True)
                logger.info("Successfully loaded EMA model state dict")
            
            # Load optimizer state - handle SAM optimizer structure
            if 'optimizer' in checkpoint and optimizer is not None:
                try:
                    optimizer_state = checkpoint['optimizer']
                    logger.info("Optimizer state will be loaded after optimizer initialization")
                except Exception as e:
                    logger.warning(f"Failed to prepare optimizer state: {e}")
                    optimizer_state = None
                
            # Load metrics from checkpoint if available
            if 'best_cer' in checkpoint:
                best_cer = checkpoint['best_cer']
            if 'best_wer' in checkpoint:
                best_wer = checkpoint['best_wer']
            if 'nb_iter' in checkpoint:
                start_iter = checkpoint['nb_iter'] + 1
                
            # Parse CER, WER, iter from filename as fallback
            m = re.search(
                r'checkpoint_(?P<cer>[\d\.]+)_(?P<wer>[\d\.]+)_(?P<iter>\d+)\.pth', checkpoint_path)
            if m and 'best_cer' not in checkpoint:
                best_cer = float(m.group('cer'))
                best_wer = float(m.group('wer'))
                start_iter = int(m.group('iter')) + 1
                
            if 'train_loss' in checkpoint:
                train_loss = checkpoint['train_loss']
            if 'train_loss_count' in checkpoint:
                train_loss_count = checkpoint['train_loss_count']
                
            # Restore random states if available (but do this after model loading)
            if 'random_state' in checkpoint:
                random.setstate(checkpoint['random_state'])
                logger.info("Restored random state")
            if 'numpy_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_state'])
                logger.info("Restored numpy random state")
            if 'torch_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_state'])
                logger.info("Restored torch random state")
            if 'torch_cuda_state' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
                logger.info("Restored torch cuda random state")
                
            # Validate that the model was loaded correctly by checking a few parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded with {total_params} total parameters")
                
            logger.info(
                f"Resumed best_cer={best_cer}, best_wer={best_wer}, start_iter={start_iter}")
        return best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count

    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = load_checkpoint(
        model, model_ema, None, getattr(args, 'resume_checkpoint', None))

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

    # Build a fixed small quick-eval split (first 256 lines) once
    try:
        from torch.utils.data import Subset
        quick_count = min(256, len(val_dataset))
        quick_indices = list(range(quick_count))
        quick_subset = Subset(val_dataset, quick_indices)
        quick_val_loader = torch.utils.data.DataLoader(quick_subset,
                                                       batch_size=args.val_bs,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       num_workers=args.num_workers)
    except Exception:
        quick_val_loader = None

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
            logger.info("Continuing training without optimizer state (will restart from initial lr/momentum)")

    # --- Helper for overlaying text on image ---
    import torchvision.transforms as T
    from PIL import Image, ImageDraw, ImageFont

    def overlay_text_on_image(img_tensor, pred_text, true_text, is_correct):
        img = T.ToPILImage()(img_tensor)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        color = "green" if is_correct else "red"
        draw.text((5, 5), f"Pred: {pred_text}", fill=color, font=font)
        draw.text((5, 25), f"GT: {true_text}", fill="blue", font=font)
        return img

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
        # Compute grad norm before first step
        loss.backward()
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm_sq += param_norm * param_norm
        grad_norm = float(total_norm_sq ** 0.5)
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
                writer.add_scalar('./Train/tone_loss', tone_loss_running / tone_count, nb_iter)
                writer.add_scalar('./Train/gate_hit_rate', gate_hit_running / tone_count, nb_iter)
                writer.add_scalar('./Train/tone_leakage', leakage_running / tone_count, nb_iter)
            # Compute EMA CTC loss for gap metric
            try:
                with torch.no_grad():
                    ema_out = model_ema.ema(image, use_masking=False)
                    if isinstance(ema_out, (list, tuple)):
                        ema_base = ema_out[0]
                    else:
                        ema_base = ema_out
                    ema_base = ema_base.float()
                    preds_size_ema = torch.IntTensor([ema_base.size(1)] * batch_size).cuda()
                    log_probs_ema = ema_base.permute(1, 0, 2).log_softmax(2)
                    torch.backends.cudnn.enabled = False
                    ctc_loss_ema = criterion(log_probs_ema, text, preds_size_ema, length).mean().item()
                    torch.backends.cudnn.enabled = True
            except Exception:
                ctc_loss_ema = 0.0

            # Build metrics dict
            metrics = {}
            metrics['loss/ctc'] = float(loss_logs.get('ctc_loss', 0.0))
            metrics['loss/ctc_ema'] = float(ctc_loss_ema)
            metrics['loss/tone'] = float(loss_logs.get('tone_loss', 0.0))
            metrics['loss/guard'] = float(loss_logs.get('guard_loss', 0.0))
            metrics['loss/total'] = float(loss.item())
            metrics['opt/lr'] = float(current_lr)
            metrics['opt/grad_norm'] = float(grad_norm)
            metrics['opt/ema_gap'] = float(metrics['loss/ctc'] - ctc_loss_ema)
            metrics['gate/hit_rate'] = float(loss_logs.get('gate_hit_rate', 0.0))
            metrics['gate/leakage'] = float(loss_logs.get('tone_leakage', 0.0))
            metrics['gate/frames_per_vowel'] = float(loss_logs.get('frames_per_vowel', 0.0))
            metrics['align/T_over_U_mean'] = float(loss_logs.get('T_over_U_mean', 0.0))
            metrics['align/T_over_U_p05'] = float(loss_logs.get('T_over_U_p05', 0.0))
            metrics['align/pct_T_over_U_lt_1p5'] = float(loss_logs.get('pct_T_over_U_lt_1p5', 0.0))
            metrics['base/blank_rate'] = float(loss_logs.get('blank_rate', 0.0))
            metrics['tone/none_rate_on_vowel'] = float(loss_logs.get('tone_none_rate_on_vowel', 0.0))
            metrics['tone/conf_margin'] = float(loss_logs.get('tone_conf_margin', 0.0))
            # Masking
            metrics['mask/ratio_effective'] = float(loss_logs.get('mask_ratio_effective', getattr(args, 'mask_ratio', 0.0)))
            metrics['mask/span_hits'] = float(loss_logs.get('mask_span_hits', 0.0))

            # Quick eval on small split
            # Run quick validation
            try:
                if quick_val_loader is None:
                    raise RuntimeError('quick_val_loader unavailable')
                model.eval()
                with torch.no_grad():
                    q_loss, q_cer, q_wer, q_preds, q_labels = valid.validation(model_ema.ema, criterion, quick_val_loader, converter, args)
                model.train()
                # Vowel CER and TER + Illegal tone rate + confusion top2 + class support from quick eval
                import editdistance
                vowel_ed_sum = 0
                vowel_len_sum = 0
                ter_err = 0
                ter_total = 0
                illegal_count = 0
                confusion = {}
                tone_support_eval = [0, 0, 0, 0, 0, 0]
                for p, g in zip(q_preds, q_labels):
                    # Vowel-only strings per gt mask
                    p_v = ''.join([p[i] for i in range(min(len(p), len(g))) if utils.is_vietnamese_vowel(g[i])])
                    g_v = ''.join([g[i] for i in range(len(g)) if utils.is_vietnamese_vowel(g[i])])
                    vowel_ed_sum += editdistance.eval(p_v, g_v)
                    vowel_len_sum += max(1, len(g_v))
                    # TER
                    L = min(len(p), len(g))
                    for i in range(L):
                        if utils.is_vietnamese_vowel(g[i]):
                            ter_total += 1
                            pt, gt = utils.tone_of_char(p[i]), utils.tone_of_char(g[i])
                            if pt != gt:
                                ter_err += 1
                                key = f"{gt}->{pt}"
                                confusion[key] = confusion.get(key, 0) + 1
                            # class support from predicted tone on vowel positions
                            tone_support_eval[pt] += 1
                    # Illegal tone rate per line
                    def has_illegal_tone(s: str) -> bool:
                        import unicodedata
                        tone_marks = {0x0301, 0x0300, 0x0309, 0x0303, 0x0323}
                        # tone on consonant
                        for ch in s:
                            d = unicodedata.normalize('NFD', ch)
                            marks = {ord(c) for c in d if unicodedata.category(c) == 'Mn' and ord(c) in tone_marks}
                            if marks and not utils.is_vietnamese_vowel(ch):
                                return True
                        # more than one tone in a syllable (word-level approximation split by space)
                        for word in s.split():
                            cnt = 0
                            for ch in word:
                                if utils.is_vietnamese_vowel(ch) and utils.tone_of_char(ch) != 0:
                                    cnt += 1
                            if cnt > 1:
                                return True
                        return False
                    if has_illegal_tone(p):
                        illegal_count += 1

                vowel_cer = float(vowel_ed_sum) / float(vowel_len_sum) if vowel_len_sum > 0 else 0.0
                ter_quick = float(ter_err) / float(max(1, ter_total))
                illegal_rate = float(illegal_count) / float(max(1, len(q_preds)))
                # confusion top2 string
                top2 = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:2]
                confusion_top2_str = ', '.join([f"{k}:{v}" for k, v in top2]) if top2 else ''
                # class support fractions from eval
                total_tone_eval = sum(tone_support_eval)
                class_support_eval = {}
                if total_tone_eval > 0:
                    names = ['NONE', 'ACUTE', 'GRAVE', 'HOOK', 'TILDE', 'DOT']
                    for i, name in enumerate(names):
                        class_support_eval[name] = tone_support_eval[i] / total_tone_eval

                metrics['eval/CER'] = float(q_cer)
                metrics['eval/WER'] = float(q_wer)
                metrics['eval/VowelCER'] = float(vowel_cer)
                metrics['eval/TER'] = float(ter_quick)
                metrics['eval/IllegalToneRate'] = float(illegal_rate)
                # Log eval class support per class
                for name, frac in class_support_eval.items():
                    metrics[f'tone/class_support/{name}'] = float(frac)
            except Exception:
                pass

            # Write scalars to TB
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'metrics/{k}', v, nb_iter)
            # Additional text logs
            if 'tone_class_support' in loss_logs and isinstance(loss_logs['tone_class_support'], dict):
                for name, frac in loss_logs['tone_class_support'].items():
                    writer.add_scalar(f'metrics/tone/class_support/{name}', frac, nb_iter)
            if 'confusion_top2_str' in locals() and confusion_top2_str:
                writer.add_text('metrics/tone/confusion_top2', confusion_top2_str, nb_iter)

            # wandb log
            log_dict = {"iter": nb_iter}
            for k, v in metrics.items():
                log_dict[f"metrics/{k}"] = v
            # Also include class support if available
            if 'tone_class_support' in loss_logs and isinstance(loss_logs['tone_class_support'], dict):
                for name, frac in loss_logs['tone_class_support'].items():
                    log_dict[f"metrics/tone/class_support/{name}"] = frac
            if 'confusion_top2_str' in locals() and confusion_top2_str:
                log_dict["metrics/tone/confusion_top2"] = confusion_top2_str
            if getattr(args, 'use_wandb', False):
                try:
                    import wandb  # type: ignore
                    wandb.log(log_dict)
                except Exception:
                    pass
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
                ter = 0.0
                try:
                    vi_pairs = [(p, g) for p, g in zip(preds, labels) if utils.contains_vietnamese(g)]
                    if len(vi_pairs) > 0:
                        tone_errors = 0
                        tone_total = 0
                        for p, g in vi_pairs:
                            L = min(len(p), len(g))
                            for i in range(L):
                                if utils.is_vietnamese_vowel(g[i]):
                                    tone_total += 1
                                    tone_errors += int(utils.tone_of_char(p[i]) != utils.tone_of_char(g[i]))
                        ter = tone_errors / max(1, tone_total)
                except Exception:
                    ter = 0.0

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
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t TER : {ter:0.4f}')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/TER', ter, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                # wandb log (optional)
                if getattr(args, 'use_wandb', False):
                    try:
                        import wandb  # type: ignore
                        # log up to 5 examples from current batch
                        example_count = min(5, batch[0].size(0))
                        example_images = []
                        # Get model predictions for current batch
                        model.eval()
                        with torch.no_grad():
                            image = batch[0].cuda()
                            # Use the same inference call as validation function (no masking for inference)
                            preds = model(image)
                            if isinstance(preds, (list, tuple)):
                                preds = preds[0]
                            preds = preds.float()
                            batch_size = image.size(0)
                            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                            preds = preds.permute(1, 0, 2).log_softmax(2)
                            _, preds_index = preds.max(2)
                            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                            preds_str = converter.decode(preds_index.data, preds_size.data)

                        for i in range(example_count):
                            img_tensor = batch[0][i].cpu()
                            pred_text = preds_str[i]
                            true_text = batch[1][i]
                            is_correct = pred_text == true_text
                            caption = f"Pred: {pred_text} | GT: {true_text} | {'✅' if is_correct else '❌'}"
                            example_images.append(wandb.Image(img_tensor, caption=caption))

                        wandb.log({
                            "val/loss": val_loss,
                            "val/CER": val_cer,
                            "val/WER": val_wer,
                            "val/best_CER": best_cer,
                            "val/best_WER": best_wer,
                            "val/examples": example_images,
                            "iter": nb_iter
                        })
                    except Exception:
                        pass
                model.train()


if __name__ == '__main__':
    main()
