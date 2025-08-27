import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import importlib
import valid
from utils import utils
from utils import vn_tags
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial
import random
import numpy as np


def compute_loss(args, model, image, batch_size, criterion, enc, autocast_ctx):
    """Compute loss for single or dual-head.
    enc: (text, length) for CTCLabelConverter OR
         (text_base, length_base, tags_mod, tags_tone, per_sample_U) for DualLabelConverter
    """
    with autocast_ctx():
        outputs = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    if not isinstance(outputs, dict):
        text, length = enc
        preds = outputs.float()
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(preds.device)
        preds = preds.permute(1,0,2).log_softmax(2)
        torch.backends.cudnn.enabled = False
        loss = criterion(preds, text.to(preds.device), preds_size, length.to(preds.device)).mean()
        torch.backends.cudnn.enabled = True
        return loss

    # Dual-head path
    text_base, length_base, tags_mod_list, tags_tone_list, _ = enc
    base = outputs['base'].float()
    mod_logits = outputs['mod'].float()
    tone_logits = outputs['tone'].float()
    device = base.device
    T = base.size(1)
    preds_size = torch.IntTensor([T]*batch_size).to(device)
    base_logp = base.permute(1,0,2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    base_loss = criterion(base_logp, text_base.to(device), preds_size, length_base.to(device)).mean()
    torch.backends.cudnn.enabled = True

    lambda_mod = args.lambda_mod
    lambda_tone = args.lambda_tone
    if lambda_mod == 0 and lambda_tone == 0:
        return base_loss

    with torch.no_grad():
        base_argmax = base.argmax(dim=2)

    # Static cache for charset & vowels
    if not hasattr(compute_loss, '_cached'):
        base_charset = args.base_charset if getattr(args, 'base_charset', None) else utils.build_base_charset()
        vowels = set('aeiouAEIOU')  # simple vowel set for weighting
        compute_loss._cached = (base_charset, vowels)
    base_charset, vowels = compute_loss._cached
    Cb = len(base_charset)

    logp_mod = torch.nn.functional.log_softmax(mod_logits, dim=2)
    logp_tone = torch.nn.functional.log_softmax(tone_logits, dim=2)

    lengths_cpu = length_base.detach().cpu().tolist()
    flat_indices = text_base.detach().cpu().tolist()
    offs=[0]
    for l in lengths_cpu:
        offs.append(offs[-1]+l)

    batch_mod_losses = torch.zeros(batch_size, device=device)
    batch_tone_losses = torch.zeros(batch_size, device=device)
    batch_mask = torch.zeros(batch_size, device=device)
    tag_alpha = args.tag_alpha
    vowel_only = args.vowel_only_tags

    # Vectorized alignment: iterate frames once per batch element using torch.unique_consecutive
    for b in range(batch_size):
        y = flat_indices[offs[b]:offs[b+1]]
        if not y:
            continue
        frames = base_argmax[b]  # [T]
        non_blank = frames != 0
        if not non_blank.any():
            continue
        # Unique consecutive collapse
        collapsed = torch.unique_consecutive(frames[non_blank])  # sequence of classes
        # Map collapsed sequence elements to GT order (greedy): find first occurrence sequentially
        gt_ptr = 0
        seg_ptr = 0
        seg_to_indices = []  # store index lists for each matched GT char
        # Build index mapping of each class in order for frames
        # Precompute run boundaries on full frame seq for averaging
        prev = torch.cat([frames[:1], frames[:-1]])
        change = frames != prev
        run_starts = torch.nonzero(change, as_tuple=True)[0]
        run_starts = torch.cat([run_starts, torch.tensor([len(frames)], device=device)])
        # Iterate runs, match to GT sequentially
        for r in range(len(run_starts)-1):
            cls = frames[run_starts[r]]
            if cls == 0:
                continue
            if gt_ptr >= len(y):
                break
            if cls.item() == y[gt_ptr]:
                seg_to_indices.append((gt_ptr, torch.arange(run_starts[r], run_starts[r+1], device=device)))
                gt_ptr += 1
        if not seg_to_indices:
            continue
        # Prepare losses
        y_mod = tags_mod_list[b].detach().cpu().tolist()
        y_tone = tags_tone_list[b].detach().cpu().tolist()
        mod_losses=[]; tone_losses=[]; weights=[]
        for gt_index, idx_tensor in seg_to_indices:
            if gt_index >= len(y):
                break
            lpm = logp_mod[b, idx_tensor, :].mean(dim=0)
            lpt = logp_tone[b, idx_tensor, :].mean(dim=0)
            # cap extreme losses to avoid inf propagating (e.g., -log(0))
            mod_loss = torch.clamp(-lpm[y_mod[gt_index]], max=60.0)
            tone_loss = torch.clamp(-lpt[y_tone[gt_index]], max=60.0)
            base_char = base_charset[y[gt_index]-1] if 1 <= y[gt_index] <= Cb else ''
            is_vowel = base_char in vowels
            if vowel_only and not is_vowel:
                continue
            weight = 1.0 if is_vowel else (1.0 if vowel_only else tag_alpha)
            mod_losses.append(mod_loss)
            tone_losses.append(tone_loss)
            weights.append(weight)
        if mod_losses:
            w = torch.tensor(weights, device=device)
            batch_mod_losses[b] = (w*torch.stack(mod_losses)).sum()/(w.sum()+1e-6)
            batch_tone_losses[b] = (w*torch.stack(tone_losses)).sum()/(w.sum()+1e-6)
            batch_mask[b]=1.0
    valid = batch_mask.sum()
    if valid>0:
        mod_loss = (batch_mod_losses*batch_mask).sum()/valid
        tone_loss = (batch_tone_losses*batch_mask).sum()/valid
    else:
        mod_loss = torch.tensor(0.0, device=device)
        tone_loss = torch.tensor(0.0, device=device)
    return base_loss + lambda_mod * mod_loss + lambda_tone * tone_loss


def main():

    args = option.get_args_parser()

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

    model = HTR_VT.create_model(
        nb_cls=args.nb_cls, img_size=args.img_size[::-1], use_dual_head=args.use_dual_head)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    if args.use_compile:
        try:
            model = torch.compile(model, mode='max-autotune')  # PyTorch 2.x
            logger.info('Model compiled with torch.compile')
        except Exception as e:
            logger.warning(f'torch.compile failed: {e}. Continuing without compile.')
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
    loader_kwargs = dict(batch_size=args.train_bs,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=args.num_workers,
                         collate_fn=partial(dataset.SameTrCollate, args=args))
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = (not args.no_persistent_workers)
        loader_kwargs['prefetch_factor'] = max(2, args.prefetch_factor)
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    vloader_kwargs = dict(batch_size=args.val_bs,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=args.num_workers)
    if args.num_workers > 0:
        vloader_kwargs['persistent_workers'] = (not args.no_persistent_workers)
        vloader_kwargs['prefetch_factor'] = max(2, args.prefetch_factor)
    val_loader = torch.utils.data.DataLoader(val_dataset, **vloader_kwargs)

    logger.info('Initializing optimizer, criterion and converter...')
    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    if args.use_dual_head:
        base_charset = args.base_charset if args.base_charset else utils.build_base_charset()
        converter = utils.DualLabelConverter(base_charset)
    else:
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
    # AMP context factory
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' and torch.cuda.is_available() else torch.float16
    def autocast_ctx():
        return torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=args.use_amp)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and args.amp_dtype=='fp16'))

    for nb_iter in range(start_iter, args.total_iter):
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        # ----- forward & backward -----
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iter)
        image = batch[0].cuda()
        enc = converter.encode(batch[1])
        batch_size = image.size(0)
        loss = compute_loss(args, model, image, batch_size, criterion, enc, autocast_ctx)
        if scaler.is_enabled():
            # First SAM step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            optimizer.first_step(zero_grad=True)
            # Second forward/backward on perturbed weights
            loss2 = compute_loss(args, model, image, batch_size, criterion, enc, autocast_ctx)
            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer)
            # Restore weights & step (second_step) using unscaled grads
            optimizer.second_step(zero_grad=True)
            scaler.update()
        else:
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, image, batch_size, criterion, enc, autocast_ctx).backward()
            optimizer.second_step(zero_grad=True)

        # NaN/inf guard
        if not torch.isfinite(loss):
            logger.error(f"Non-finite loss detected (loss={loss.item()}). Aborting iteration {nb_iter}.")
            break
        model.zero_grad(set_to_none=True)
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()
        train_loss_count += 1

        # ----- logging -----
        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0
            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f}')
            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            if wandb is not None:
                wandb.log({"train/lr": current_lr, "train/loss": train_loss_avg, "iter": nb_iter}, step=nb_iter)
            train_loss = 0.0
            train_loss_count = 0

        # ----- evaluation -----
        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(
                    model_ema.ema, criterion, val_loader, converter, args)

                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!')
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

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                # wandb log (optional): examples and metrics (table style)
                if wandb is not None:
                    # log up to 5 examples from current batch
                    example_count = min(5, batch[0].size(0))
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
                        preds_size = torch.IntTensor(
                            [preds.size(1)] * batch_size)
                        preds = preds.permute(1, 0, 2).log_softmax(2)
                        _, preds_index = preds.max(2)
                        preds_index = preds_index.transpose(
                            1, 0).contiguous().view(-1)
                        preds_str = converter.decode(
                            preds_index.data, preds_size.data)

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
                        "val/loss": val_loss,
                        "val/CER": val_cer,
                        "val/WER": val_wer,
                        "val/best_CER": best_cer,
                        "val/best_WER": best_wer,
                        "val/examples_table": examples_table,
                        "iter": nb_iter
                    }, step=nb_iter)
                model.train()


if __name__ == '__main__':
    main()
