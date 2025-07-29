#!/bin/bash

# Training script for Encoder-Decoder HTR model on IAM dataset

echo "Training HTR Encoder-Decoder model on IAM dataset..."

python train_encoder_decoder.py IAM \
    --out-dir "./output" \
    --exp-name "IAM_HTR_EncoderDecoder" \
    --train-bs 8 \
    --val-bs 1 \
    --num-workers 4 \
    --eval-iter 1000 \
    --total-iter 100000 \
    --warm-up-iter 1000 \
    --print-iter 100 \
    --max-lr 1e-3 \
    --weight-decay 0.05 \
    --img-size 512 64 \
    --patch-size 4 32 \
    --mask-ratio 0.0 \
    --max-span-length 4 \
    --proj 8 \
    --ema-decay 0.9999 \
    --seed 123 \
    --train-data-list "./data/iam/train.ln" \
    --data-path "./data/iam/lines/" \
    --val-data-list "./data/iam/val.ln" \
    --test-data-list "./data/iam/test.ln" \
    --nb-cls 80 \
    --decoder-layers 6 \
    --decoder-heads 8 \
    --max-seq-len 256 \
    --label-smoothing 0.1 \
    --beam-size 5

echo "Training completed!"
