#!/bin/bash
# Exp 1: Impact of Classifier-Free Guidance Weight
# Adjust CKPT to your best model path
CKPT="results/train_cross_attn/best.pth"

for cfg in 1.0 2.0 3.0 5.0 8.0
do
    echo "========================================"
    echo "Sampling with CFG weight: $cfg"
    echo "========================================"
    python sample.py --ckpt-path $CKPT --cfg_weight $cfg --scheduler_type ddpm --num_test_timesteps 1000
done
