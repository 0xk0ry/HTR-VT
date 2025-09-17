import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from model.sgm import *
from functools import partial
import random
import numpy as np
import re
import importlib


def compute_loss(args, model, criterion, batch, converter,
                 sgm_on=False, sgm_head=None, sgm_vocab_stoi=None,
                 nb_iter=0):
    """
    Returns total_loss, ctc_loss_detached, sgm_loss_detached
    Uses one encoder pass and shares features across CTC & SGM.
    """
    images = batch[0].cuda()
    labels_raw = batch[1]                     # list[str]
    text, length = converter.encode(labels_raw)
    batch_size = images.size(0)

    # ---- encoder features once ----
    feats = model.extract_features(
        # (B,L,D)
        images, args.mask_ratio, args.max_span_length, use_masking=True)

    # ---- CTC loss ----
    logits = model.head(feats).float()       # (B,L,V_ctc)
    preds_size = torch.IntTensor([logits.size(1)] * batch_size).cuda()
    log_probs = logits.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    ctc_loss = criterion(log_probs, text.cuda(),
                         preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True

    # ---- SGM (optional; train-time) ----
    sgm_loss = torch.tensor(0., device=logits.device)
    warmup_iters = getattr(args, "sgm_warmup_iters", 0)
    if sgm_on and (sgm_head is not None) and (nb_iter >= warmup_iters):
        txt_ids, tgt_ids, mask_pos = make_sgm_batch(
            labels_raw, sgm_vocab_stoi, getattr(args, "sgm_mask_rate", 0.15))
        txt_ids, tgt_ids, mask_pos = txt_ids.cuda(), tgt_ids.cuda(), mask_pos.cuda()
        sgm_logits, _ = sgm_head(feats, txt_ids, mask_pos)  # (B,N,V)
        sgm_logits = sgm_logits.view(-1, sgm_logits.size(-1))
        tgt_flat = tgt_ids.view(-1)
        sgm_loss = F.cross_entropy(sgm_logits, tgt_flat, ignore_index=-100)

    total_loss = ctc_loss + getattr(args, "sgm_lambda", 0.15) * sgm_loss
    return total_loss, ctc_loss.detach(), sgm_loss.detach()


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
            logger.warning(
                f"Failed to initialize wandb: {e}. Continuing without wandb.")
            wandb = None
    else:
        wandb = None

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

    # Use centralized checkpoint loader like model_v4-2
    resume_path = args.resume if getattr(
        args, 'resume', None) else getattr(args, 'resume_checkpoint', None)
    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = utils.load_checkpoint(
        model, model_ema, None, resume_path, logger)

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

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    sgm_on = args.sgm_on
    sgm_vocab_itos, sgm_vocab_stoi = build_sgm_vocab(converter.dict.keys())
    sgm_head = (SGMHead(vocab_size=len(sgm_vocab_itos),
                        d_vis=model.head.in_features,
                        d_sgm=args.sgm_dmodel,
                        num_layers=args.sgm_layers,
                        num_heads=args.sgm_heads,
                        dropout=args.sgm_dropout).cuda()
                if sgm_on else None)

    params = list(model.parameters())
    if sgm_on:
        params += list(sgm_head.parameters())
    optimizer = sam.SAM(params, torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)

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

        # 1st SAM step
        total_loss, ctc_loss_val, sgm_loss_val = compute_loss(
            args, model, criterion, batch, converter,
            sgm_on=sgm_on, sgm_head=sgm_head, sgm_vocab_stoi=sgm_vocab_stoi,
            nb_iter=nb_iter
        )
        total_loss.backward()
        optimizer.first_step(zero_grad=True)

        # 2nd SAM step: recompute the SAME loss with perturbed weights
        total_loss, ctc_loss_val, sgm_loss_val = compute_loss(
            args, model, criterion, batch, converter,
            sgm_on=sgm_on, sgm_head=sgm_head, sgm_vocab_stoi=sgm_vocab_stoi,
            nb_iter=nb_iter
        )
        total_loss.backward()
        optimizer.second_step(zero_grad=True)

        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)

        train_loss += total_loss.item()
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} '
                        f'\t train_loss : {train_loss_avg:0.5f} '
                        f'\t CTC : {ctc_loss_val.item():0.5f} '
                        f'\t SGM : {sgm_loss_val.item():0.5f}')

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            if wandb is not None:
                wandb.log({
                    'train/lr': current_lr,
                    'train/loss': train_loss_avg,
                    'train/ctc_loss': ctc_loss_val.item(),
                    'train/sgm_loss': sgm_loss_val.item(),
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
