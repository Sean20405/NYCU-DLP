#!/bin/bash
# Exp 2: Sampling Efficiency (DDPM vs DDIM)
CKPT="results/train_cross_attn/best.pth"

# Test DDIM with different steps
for steps in 10 25 50 100
do
    echo "========================================"
    echo "Running DDPM with $steps steps"
    echo "========================================"
    python sample.py --ckpt-path $CKPT --scheduler_type ddpm --num_test_timesteps $steps
done

# Test DDPM (Baseline)
echo "========================================"
echo "Running DDPM with 1000 steps"
echo "========================================"
python sample.py --ckpt-path $CKPT --scheduler_type ddpm --num_test_timesteps 1000
