import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import wandb
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial
import random
import numpy as np


def compute_loss(args, model, image, batch_size, criterion, text, length, diacritic_targets):
    base_logits, diacritic_logits = model(image, args.mask_ratio,
                                          args.max_span_length, use_masking=True)
    # base_logits: (B, T, nb_cls), diacritic_logits: (B, T, 6)
    # CTC loss for base chars
    base_logits = base_logits.float()
    preds_size = torch.IntTensor([base_logits.size(1)] * batch_size).cuda()
    base_logits_ctc = base_logits.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    loss_ctc = criterion(base_logits_ctc, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True

    # Cross-entropy loss for diacritics
    diacritic_logits_flat = diacritic_logits.view(-1, 6)
    diacritic_targets_flat = diacritic_targets.view(-1).cuda()
    loss_diac = torch.nn.functional.cross_entropy(diacritic_logits_flat, diacritic_targets_flat)

    # Hybrid loss
    loss = args.alpha_ctc * loss_ctc + args.alpha_diac * loss_diac
    return loss, loss_ctc.item(), loss_diac.item()


def main():

    args = option.get_args_parser()

    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    # Initialize wandb
    wandb.init(project="HTR-VT", name=args.exp_name+"diacritic",
               config=vars(args), dir=args.save_dir)

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
            
            # Handle checkpoint compatibility between old single-task and new multi-task models
            try:
                model.load_state_dict(model_dict, strict=True)
                logger.info("Successfully loaded main model state dict (exact match)")
            except RuntimeError as e:
                if "Missing key(s)" in str(e) and ("base_head" in str(e) or "diacritic_head" in str(e)):
                    logger.warning("Checkpoint appears to be from old single-task model. Attempting partial loading...")
                    
                    # Try to convert old head to new base_head
                    if 'head.weight' in model_dict and 'head.bias' in model_dict:
                        model_dict['base_head.weight'] = model_dict.pop('head.weight')
                        model_dict['base_head.bias'] = model_dict.pop('head.bias')
                        logger.info("Converted old 'head' to 'base_head'")
                    
                    # Initialize diacritic_head randomly if not present
                    current_state = model.state_dict()
                    if 'diacritic_head.weight' not in model_dict:
                        model_dict['diacritic_head.weight'] = current_state['diacritic_head.weight']
                        model_dict['diacritic_head.bias'] = current_state['diacritic_head.bias']
                        logger.info("Initialized diacritic_head with random weights")
                    
                    # Try loading again
                    model.load_state_dict(model_dict, strict=True)
                    logger.info("Successfully loaded converted model state dict")
                else:
                    # Re-raise if it's a different error
                    raise e
            
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
                                               collate_fn=dataset.MultiTaskCollate)
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             collate_fn=dataset.MultiTaskCollate)

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
    for nb_iter in range(start_iter, args.total_iter):
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        base_texts = batch[1]
        diacritic_targets = torch.tensor(batch[2], dtype=torch.long)  # (B, T)
        text, length = converter.encode(base_texts)
        batch_size = image.size(0)
        loss, loss_ctc, loss_diac = compute_loss(args, model, image, batch_size,
                                                criterion, text, length, diacritic_targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss_second, _, _ = compute_loss(args, model, image, batch_size,
                                        criterion, text, length, diacritic_targets)
        loss_second.backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            logger.info(f'Iter {nb_iter:6d} | Loss: {train_loss/train_loss_count:.4f} | '
                       f'CTC Loss: {loss_ctc:.4f} | Diac Loss: {loss_diac:.4f} | LR: {current_lr:.6f}')
            # Log to wandb/tensorboard
            if args.use_wandb:
                wandb.log({
                    'train/loss': train_loss/train_loss_count,
                    'train/loss_ctc': loss_ctc,
                    'train/loss_diac': loss_diac,
                    'train/lr': current_lr
                }, step=nb_iter)
            else:
                writer.add_scalar('train/loss', train_loss/train_loss_count, nb_iter)
                writer.add_scalar('train/loss_ctc', loss_ctc, nb_iter)
                writer.add_scalar('train/loss_diac', loss_diac, nb_iter)
                writer.add_scalar('train/lr', current_lr, nb_iter)

        if nb_iter % args.eval_iter == 0:
            model.eval()
            model_ema.ema.eval()
            
            with torch.no_grad():
                val_loss, val_cer, val_wer, _, _, val_diac_acc = valid.validation(model_ema.ema,
                                                                                 criterion,
                                                                                 val_loader,
                                                                                 converter)
            
            logger.info(f'Validation | Loss: {val_loss:.4f} | CER: {val_cer:.4f} | '
                       f'WER: {val_wer:.4f} | Diac Acc: {val_diac_acc:.4f}')
            
            # Log validation metrics
            if args.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/cer': val_cer,
                    'val/wer': val_wer,
                    'val/diacritic_acc': val_diac_acc
                }, step=nb_iter)
            else:
                writer.add_scalar('val/loss', val_loss, nb_iter)
                writer.add_scalar('val/cer', val_cer, nb_iter)
                writer.add_scalar('val/wer', val_wer, nb_iter)
                writer.add_scalar('val/diacritic_acc', val_diac_acc, nb_iter)
            
            # Save best model based on CER
            if val_cer < best_cer:
                best_cer = val_cer
                best_wer = val_wer
                logger.info(f'New best CER: {best_cer:.4f}, WER: {best_wer:.4f}, Diac Acc: {val_diac_acc:.4f}')
                
                # Save checkpoint
                torch.save({
                    'model': model.state_dict(),
                    'state_dict_ema': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_cer': best_cer,
                    'best_wer': best_wer,
                    'best_diac_acc': val_diac_acc,
                    'iter': nb_iter,
                    'args': args
                }, os.path.join(args.save_dir, 'best_CER.pth'))
                
                # Also save with iteration info
                torch.save({
                    'model': model.state_dict(),
                    'state_dict_ema': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_cer': best_cer,
                    'best_wer': best_wer,
                    'best_diac_acc': val_diac_acc,
                    'iter': nb_iter,
                    'args': args
                }, os.path.join(args.save_dir, f'checkpoint_{best_cer:.4f}_{best_wer:.4f}_{nb_iter}.pth'))
            
            model.train()
            train_loss, train_loss_count = 0.0, 0
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(args, model, image, batch_size,
                     criterion, text, length).backward()
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
            wandb.log({"train/lr": current_lr,
                      "train/loss": train_loss_avg, "iter": nb_iter})
            train_loss = 0.0
            train_loss_count = 0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)

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
                # wandb log
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
                    preds_index = preds_index.transpose(
                        1, 0).contiguous().view(-1)
                    preds_str = converter.decode(
                        preds_index.data, preds_size.data)

                for i in range(example_count):
                    img_tensor = batch[0][i].cpu()
                    pred_text = preds_str[i]
                    true_text = batch[1][i]
                    is_correct = pred_text == true_text
                    caption = f"Pred: {pred_text} | GT: {true_text} | {'✅' if is_correct else '❌'}"
                    example_images.append(wandb.Image(
                        img_tensor, caption=caption))

                wandb.log({
                    "val/loss": val_loss,
                    "val/CER": val_cer,
                    "val/WER": val_wer,
                    "val/best_CER": best_cer,
                    "val/best_WER": best_wer,
                    "val/examples": example_images,
                    "iter": nb_iter
                })
                model.train()


if __name__ == '__main__':
    main()
