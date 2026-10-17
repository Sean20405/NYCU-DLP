#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, out_dim)
        self.log_std = nn.Linear(128, out_dim)
        
        init_layer_uniform(self.mu)
        init_layer_uniform(self.log_std)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        mu = torch.tanh(self.mu(x)) * 2.0
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist

def main():
    parser = argparse.ArgumentParser(description="Test Task 3 PPO Model")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint (.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint and print details
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path '{args.model_path}' does not exist.")

    # If input is dir, switch to evaluate all ckpt
    if os.path.isdir(args.model_path):
        ckpt_paths = glob.glob(os.path.join(args.model_path, "task3_*.pth"))
        print(f"Evaluate {len(ckpt_paths)} step checkpoint from {args.model_path}")
        

    for ckpt_path in ckpt_paths:
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Print checkpoint details
        print("=" * 40)
        print("Checkpoint Details:")
        for key, val in ckpt.items():
            if key not in ['actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'actor_scheduler', 'critic_scheduler']:
                print(f"  {key}: {val}")
        print("=" * 40)

        # Initialize environment and Actor network
        env = gym.make("Walker2d-v5")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        actor = Actor(obs_dim, action_dim).to(device)
        
        if isinstance(ckpt, dict) and 'actor' in ckpt:
            actor.load_state_dict(ckpt['actor'])
        else:
            actor.load_state_dict(ckpt)
        
        actor.eval()

        # Retrieve obs_rms if present in checkpoint
        obs_rms = None
        if isinstance(ckpt, dict) and 'obs_rms' in ckpt and ckpt['obs_rms'] is not None:
            obs_rms = ckpt['obs_rms']
            print("[Info] Loaded running mean & std statistics (obs_rms) from checkpoint.")
        else:
            print("[Warning] No observation normalization statistics found in checkpoint. Evaluating with raw states.")

        # Run inference for 20 seeds (0 to 19)
        returns = []
        for seed in range(20):
            state, _ = env.reset(seed=seed)
            done = False
            total_reward = 0.0
            
            while not done:
                # Apply observation normalization if statistics are loaded
                if obs_rms is not None:
                    state = (state - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
                    state = np.clip(state, -10.0, 10.0)

                state_t = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    _, dist = actor(state_t)
                    action = dist.mean.clamp(-2.0, 2.0).cpu().numpy()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
            
            print(f"Seed {seed:02d}: Return = {total_reward:.4f}")
            returns.append(total_reward)

        avg_return = np.mean(returns)
        # Print average return in bold green
        print("-" * 40)
        print(f"\033[1;32mAverage Return: {avg_return:.4f}\033[0m")
        print("-" * 40)
        
        env.close()

if __name__ == "__main__":
    main()
