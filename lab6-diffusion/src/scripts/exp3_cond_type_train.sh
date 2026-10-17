#!/bin/bash
# Exp 3: Conditioning Mechanism Comparison (Training)
# Note: These will take time as they run full training

echo "========================================"
echo "Training with Concat conditioning"
echo "========================================"
# python main.py --cond_type concat --exp-name "train_concat" --epochs 200
python sample.py --ckpt-path results/train_concat/best.pth --cond_type concat --exp-name "eval_train_concat" --scheduler_type ddpm --num_test_timesteps 1000

echo "========================================"
echo "Training with Cross-Attention conditioning"
echo "========================================"
# python main.py --cond_type cross_attn --exp-name "train_cross_attn" --epochs 200
python sample.py --ckpt-path results/train_cross_attn/best.pth --cond_type cross_attn --exp-name "eval_train_cross_attn" --scheduler_type ddpm --num_test_timesteps 1000
