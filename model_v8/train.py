import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

import os
import json
import importlib
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial
import random
import numpy as np
import time


def compute_loss(args, model, image, batch_size, criterion, enc):
    """Unified loss: base CTC + optional tone CE (frame->segment alignment)."""
    outputs = model(image, args.mask_ratio,
                    args.max_span_length, use_masking=True)
    if not isinstance(outputs, dict):  # fallback single head
        text, length = enc
        preds = outputs.float()
        preds_size = torch.full((batch_size,), preds.size(
            1), dtype=torch.int32, device=preds.device)
        logp = preds.permute(1, 0, 2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        loss = criterion(logp, text.to(preds.device),
                         preds_size, length.to(preds.device)).mean()
        torch.backends.cudnn.enabled = True
        return loss

    text_base, length_base, tone_lists, _ = enc
    base_logits = outputs['base'].float()
    tone_logits = outputs['tone'].float()
    device = base_logits.device
    T = base_logits.size(1)
    preds_size = torch.full((batch_size,), T, dtype=torch.int32, device=device)
    base_logp = base_logits.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    base_loss = criterion(base_logp, text_base.to(
        device), preds_size, length_base.to(device)).mean()
    torch.backends.cudnn.enabled = True

    lambda_tone = getattr(args, 'lambda_tone', 1.0)
    if lambda_tone == 0.0:
        return base_loss

    # Tone loss: align collapsed argmax segments to targets.
    with torch.no_grad():
        frame_argmax = base_logits.argmax(dim=2)  # [B,T]
    p_tone = torch.softmax(tone_logits, dim=2)              # (B,T,6)
    P_base = base_logp.permute(1, 0, 2).exp()               # (B,T,C_base)
    batch_losses = []
    batch_weights = []
    offset = 0
    base_chars = ''.join(utils.build_base_charset())
    vowel_set = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY')
    alpha_consonant = 0.2  # small weight for consonants
    for b in range(batch_size):
        U = int(length_base[b].item())
        if U == 0:
            continue
        target_seq = text_base[offset:offset+U].tolist()  # GT base class ids (1..)
        tone_target = tone_lists[b].to(device)            # GT tone ids per base char
        offset += U

        # Frame-level predicted base ids
        frames = frame_argmax[b]
        non_blank = frames.ne(0)
        prev = torch.cat([frames.new_zeros(1), frames[:-1]])
        change = frames.ne(prev) & non_blank
        starts = torch.nonzero(change, as_tuple=True)[0]
        if starts.numel() == 0:
            continue

        # Build segments (class id, frame indices)
        seg_vals = frames[starts]
        segments = []
        for si in range(starts.numel()):
            st = starts[si]
            en = starts[si+1] if si+1 < starts.numel() else frames.numel()
            segments.append((int(seg_vals[si].item()), torch.arange(st, en, device=device)))

        seg_ptr = 0
        char_losses = []
        vowel_mask = []  # 1 for vowel, 0 for consonant

        for u, cls in enumerate(target_seq):
            # advance to matching segment; if not found, skip (do not break to keep later chars)
            while seg_ptr < len(segments) and segments[seg_ptr][0] != cls:
                seg_ptr += 1
            if seg_ptr >= len(segments):
                # no further matching segment; stop mapping
                continue
            _, idxs = segments[seg_ptr]
            seg_ptr += 1
            # gamma-lite weights inside segment
            cls_prob_frames = P_base[b, idxs, cls]                  # (S,)
            blank_prob_frames = P_base[b, idxs, 0]                  # (S,)
            w = (1.0 - blank_prob_frames) * cls_prob_frames         # (S,)
            if w.sum() <= 0:
                continue
            w = w / (w.sum() + 1e-6)
            seg_p_tone = (p_tone[b, idxs, :] * w.unsqueeze(1)).sum(dim=0)  # (6,)
            tone_id = tone_target[u]
            char_losses.append(-torch.log(seg_p_tone[tone_id] + 1e-8))
            ch = base_chars[cls-1] if 1 <= cls <= len(base_chars) else ''
            vowel_mask.append(1.0 if ch in vowel_set else 0.0)

        if char_losses:
            lc = torch.stack(char_losses)                                # [M]
            vm = torch.tensor(vowel_mask, device=device)                 # [M]
            weights = vm + alpha_consonant * (1.0 - vm)                  # vowel=1, consonant=alpha
            batch_losses.append((lc * weights).sum() / (weights.sum() + 1e-6))
    tone_loss = torch.stack(batch_losses).mean(
    ) if batch_losses else torch.tensor(0.0, device=device)
    return base_loss + lambda_tone * tone_loss


def main():

    args = option.get_args_parser()

    torch.backends.cudnn.benchmark = True          # autotune convs
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 speed on Ampere+
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    except Exception:
        pass

    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    # Initialize wandb (optional, model_v4-2 style)
    if getattr(args, 'use_wandb', False):
        try:
            wandb = importlib.import_module('wandb')
            wandb.init(project=getattr(args, 'wandb_project', 'HTR-VT'), name=args.exp_name,
                       config=vars(args), dir=args.save_dir)
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(
                f"Failed to initialize wandb: {e}. Continuing without wandb.")
            wandb = None
    else:
        wandb = None

    # Always create model with tone head for v8 tone architecture
    base_charset_str = utils.build_base_charset()
    nb_cls = len(base_charset_str) + 1
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=args.img_size[::-1])

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
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)

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
                logger.info(
                    "Loading main model from 'state_dict_ema' (fallback)")
            else:
                raise KeyError(
                    "Neither 'model' nor 'state_dict_ema' found in checkpoint")

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
                    logger.info(
                        "Optimizer state will be loaded after optimizer initialization")
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
                                               persistent_workers=True,
                                               prefetch_factor=4,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )

    logger.info('Initializing optimizer, criterion and converter...')
    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.ToneLabelConverter(base_charset_str)

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
    use_amp = getattr(args, 'amp', False) and torch.cuda.is_available()
    scaler = GradScaler('cuda')
    logger.info('Start training...')
    for nb_iter in range(start_iter, args.total_iter):
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        # (text_base, length_base, tone_lists, duplicate_len)
        enc = converter.encode(batch[1])
        batch_size = image.size(0)
        if use_amp:
            with autocast('cuda'):
                loss = compute_loss(args, model, image,
                                    batch_size, criterion, enc)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer.base_optimizer)
            optimizer.first_step(zero_grad=True)
            with autocast('cuda'):
                loss_second = compute_loss(
                    args, model, image, batch_size, criterion, enc)
            scaler.scale(loss_second).backward()
            # Unscale gradients before SAM second step weight restore + base optimizer update
            scaler.unscale_(optimizer.base_optimizer)
            # Restore weights and apply base optimizer step through SAM second_step logic
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state_p = optimizer.state.get(p, {})
                    if "old_p" in state_p:
                        p.data = state_p["old_p"]
                        # match SAM.second_step behavior (keeps old_p) but we can delete to save memory
                        del state_p["old_p"]
            scaler.step(optimizer.base_optimizer)
            scaler.update()
            optimizer.zero_grad()
            model.zero_grad()
        else:
            loss = compute_loss(args, model, image, batch_size, criterion, enc)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, image, batch_size,
                         criterion, enc).backward()
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
            # wandb log
            if wandb is not None:
                wandb.log({"train/lr": current_lr,
                          "train/loss": train_loss_avg, "iter": nb_iter}, step=nb_iter)

            train_loss = 0.0
            train_loss_count = 0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_ret = valid.validation(model_ema.ema,
                                           criterion,
                                           val_loader,
                                           converter,
                                           args)
                (val_loss_base,
                 val_loss_full,
                 val_cer_base,
                 val_wer_base,
                 val_cer,
                 val_wer,
                 val_ter,
                 preds,
                 labels) = val_ret

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
                    f'Val. loss_base : {val_loss_base:0.3f} \t loss_full : {val_loss_full:0.3f} \t '
                    f'CER_base : {val_cer_base:0.4f} \t WER_base : {val_wer_base:0.4f} \t '
                    f'CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t TER : {val_ter:0.4f}')

                # Save checkpoint every print interval
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

                writer.add_scalar('./VAL/CER_base', val_cer_base, nb_iter)
                writer.add_scalar('./VAL/WER_base', val_wer_base, nb_iter)
                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/TER', val_ter, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss_base', val_loss_base, nb_iter)
                writer.add_scalar('./VAL/val_loss_full', val_loss_full, nb_iter)
                # wandb log (optional): examples and metrics (table style)
                if wandb is not None:
                    # log up to 5 examples from current batch
                    example_count = min(5, batch[0].size(0))
                    # Get model predictions for current batch
                    model.eval()
                    with torch.no_grad():
                        image = batch[0].cuda()
                        # Use the same inference call as validation function (no masking for inference)
                        outputs = model(image)
                        if isinstance(outputs, dict):
                            base_logits = outputs['base'].float()
                            tone_logits = outputs.get('tone', None)
                        else:
                            base_logits = outputs.float()
                            tone_logits = None
                        B = image.size(0)
                        preds_size = torch.IntTensor([base_logits.size(1)] * B)
                        base_logp = base_logits.permute(1, 0, 2).log_softmax(2)
                        _, base_idx = base_logp.max(2)  # [T,B]
                        base_idx_flat = base_idx.transpose(1, 0).contiguous().view(-1)
                        base_strs = converter.decode(base_idx_flat.data, preds_size.data)
                        if tone_logits is not None:
                            tone_frame = tone_logits.argmax(dim=2)  # [B,T]
                            tone_char_preds = []
                            for b in range(B):
                                fcls = base_logits[b].argmax(dim=1)
                                tcls = tone_frame[b]
                                prev = -1
                                tc = []
                                for fc, tcid in zip(fcls.tolist(), tcls.tolist()):
                                    if fc != 0 and fc != prev:
                                        tc.append(tcid)
                                        prev = fc
                                tone_char_preds.extend(tc)
                            tone_char_preds_tensor = torch.tensor(tone_char_preds, device=base_logits.device)
                            try:
                                preds_str = converter.decode_with_tones(base_idx_flat.data, preds_size.data, tone_char_preds_tensor)
                            except Exception:
                                preds_str = base_strs
                        else:
                            preds_str = base_strs

                    examples_table = wandb.Table(
                        columns=["iter", "index", "image", "pred", "gt", "correct"])
                    for i in range(example_count):
                        img_tensor = batch[0][i].cpu()
                        pred_text = preds_str[i]
                        true_text = batch[1][i]
                        is_correct = pred_text == true_text
                        examples_table.add_data(nb_iter, i, wandb.Image(
                            img_tensor), pred_text, true_text, bool(is_correct))

                    wandb.log({
                        "val/loss_base": val_loss_base,
                        "val/loss_full": val_loss_full,
                        "val/CER_base": val_cer_base,
                        "val/WER_base": val_wer_base,
                        "val/CER": val_cer,
                        "val/WER": val_wer,
                        "val/TER": val_ter,
                        "val/best_CER": best_cer,
                        "val/best_WER": best_wer,
                        "val/examples_table": examples_table,
                        "iter": nb_iter
                    }, step=nb_iter)
                model.train()


if __name__ == '__main__':
    main()
