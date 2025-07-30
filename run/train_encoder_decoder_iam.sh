#!/bin/bash

# Training script for Encoder-Decoder HTR model on IAM dataset

echo "Training HTR Encoder-Decoder model on IAM dataset..."

!python HTR-VT/train_encoder_decoder.py --exp-name "iam_encoder_decoder" --eval-iter 500 --max-lr 1e-3 --train-bs 32 --val-bs 1 --weight-decay 0.05 --img-size 512 64 --proj 8 --dila-ero-max-kernel 2 --dila-ero-iter 1 --proba 0.5 --total-iter 100000 --warm-up-iter 1000 --eval-iter 1000 --print-iter 100 --num-workers 4 --decoder-layers 4 --decoder-heads 6 --max-seq-len 256 --label-smoothing 0.1 --beam-size 5 --generation-method nucleus --generation-temperature 0.7 --repetition-penalty 1.3 --top-p 0.9 --ema-decay 0.9999 --seed 42 IAM --data-path "/kaggle/input/iam-vt/lines/" --train-data-list "/kaggle/working/HTR-VT/data/iam/train.ln" --val-data-list "/kaggle/working/HTR-VT/data/iam/val.ln" --test-data-list "/kaggle/working/HTR-VT/data/iam/test.ln" --nb-cls 80

echo "Training completed!"
