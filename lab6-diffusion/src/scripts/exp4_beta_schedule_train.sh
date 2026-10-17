#!/bin/bash
# Exp 4: Beta Schedule Comparison (Training)

for schedule in "linear" "scaled_linear" "squaredcos_cap_v2"
do
    echo "========================================"
    echo "Training with beta schedule: $schedule"
    echo "========================================"
    # python main.py --beta_schedule $schedule --exp-name "train_beta_$schedule" --epochs 200
    python sample.py --ckpt-path "results/train_beta_$schedule/best.pth" --beta_schedule $schedule --exp-name "eval_train_beta_$schedule"
done
