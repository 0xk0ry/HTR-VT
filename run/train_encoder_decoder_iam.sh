#!/bin/bash

# Training script for Encoder-Decoder HTR model on IAM dataset

echo "Training HTR Encoder-Decoder model on IAM dataset..."

python train_encoder_decoder.py IAM --exp-name "iam_encoder_decoder" --max-lr 1e-3 --train-bs 8 --val-bs 1 --weight-decay 0.05 --img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 --proba 0.5 --total-iter 100000 --warm-up-iter 1000 --eval-iter 1000 --print-iter 100 --num-workers 4 --data-path "/kaggle/input/iam-vt/lines/" --train-data-list "/kaggle/working/HTR-VT/data/iam/train.ln" --val-data-list "/kaggle/working/HTR-VT/data/iam/val.ln" --test-data-list "/kaggle/working/HTR-VT/data/iam/test.ln" --nb-cls 80 --decoder-layers 4 --decoder-heads 6 --max-seq-len 256 --label-smoothing 0.1 --beam-size 5 --ema-decay 0.9999 --seed 123

echo "Training completed!"
