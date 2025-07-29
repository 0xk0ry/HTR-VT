#!/bin/bash

# Test script for Encoder-Decoder HTR model on IAM dataset

echo "Testing HTR Encoder-Decoder model on IAM dataset..."

python test_encoder_decoder.py \
    --checkpoint "./output/IAM_HTR_EncoderDecoder_encoder_decoder/best_CER.pth" \
    --test-data-list "./data/iam/test.ln" \
    --data-path "./data/iam/lines/" \
    --img-size 512 64 \
    --max-length 256 \
    --batch-size 1 \
    --method "beam_search" \
    --output-dir "./test_results/iam_encoder_decoder"

echo "Testing completed!"
