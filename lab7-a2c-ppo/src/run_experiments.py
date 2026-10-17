#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import time
import os

# Define configurations to run: (exp_name, epsilon, entropy_weight, cuda_device)
# We test 4 clipping parameter values (with entropy_weight=0.01)
# and 4 entropy weight values (with epsilon=0.2)
# Since the default baseline (epsilon=0.2, entropy_weight=0.01) is shared,
# we only need to run 7 configurations total.
configs = [
    ("clip_0.05", 0.05, 0.01, "1"),
    ("clip_0.2_default", 0.2, 0.01, "1"),
    ("clip_0.5", 0.5, 0.01, "2"),
    ("clip_1.0", 1.0, 0.01, "2"),
    
    ("entropy_0.0", 0.2, 0.0, "1"),
    ("entropy_0.001", 0.2, 0.001, "2"),
    ("entropy_0.1", 0.2, 0.1, "1"),
]

python_bin = sys.executable
script_path = "ppo_walker-3.py"
log_dir = "results"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Run maximum of 4 parallel processes at a time to prevent any CPU/GPU congestion
max_parallel = 4
active_processes = []

print("Starting PPO Empirical Study Hyperparameter Grid Search on Walker2d-v5...")
print("Assigning runs to idle GPUs (GPU 1 and GPU 2)...")

for exp_name, eps, ent, gpu in configs:
    # Wait if we already have max_parallel processes running
    while len(active_processes) >= max_parallel:
        # Check status of running processes
        still_running = []
        for p, name in active_processes:
            ret = p.poll()
            if ret is None:
                still_running.append((p, name))
            else:
                print(f"  [+] Finished configuration: {name} (Exit code: {ret})")
        active_processes = still_running
        if len(active_processes) >= max_parallel:
            time.sleep(10)
            
    # Launch new process
    cmd = [
        python_bin, script_path,
        "--num-episodes", "1500",
        "--epsilon", str(eps),
        "--entropy-weight", str(ent),
        "--exp-name", exp_name,
        "--log-dir", log_dir,
        "--obs-norm",
        "--adv-norm",
        "--sched"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["WANDB_MODE"] = "offline"
    env["LD_LIBRARY_PATH"] = ""
    
    stdout_log = open(f"logs/{exp_name}_stdout.log", "w")
    stderr_log = open(f"logs/{exp_name}_stderr.log", "w")
    
    print(f"  [>] Launching {exp_name} on GPU {gpu} (epsilon={eps}, entropy={ent})...")
    p = subprocess.Popen(cmd, env=env, stdout=stdout_log, stderr=stderr_log, text=True)
    active_processes.append((p, exp_name))
    time.sleep(2) # stagger launches slightly

# Wait for all remaining processes to finish
while active_processes:
    still_running = []
    for p, name in active_processes:
        ret = p.poll()
        if ret is None:
            still_running.append((p, name))
        else:
            print(f"  [+] Finished configuration: {name} (Exit code: {ret})")
    active_processes = still_running
    if active_processes:
        time.sleep(10)

print("\nAll empirical study experiments completed successfully!")
