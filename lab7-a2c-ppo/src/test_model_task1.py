#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, out_dim)
        self.log_std = nn.Linear(128, out_dim)
        
        initialize_uniformly(self.mu)
        initialize_uniformly(self.log_std)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        mu = torch.tanh(self.mu(x)) * 2.0
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist

def main():
    parser = argparse.ArgumentParser(description="Test Task 1 A2C Model")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint (.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint and print details
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path '{args.model_path}' does not exist.")

    print(f"Loading checkpoint from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Print checkpoint details
    print("=" * 40)
    print("Checkpoint Details:")
    for key, val in ckpt.items():
        if key not in ['actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'actor_scheduler', 'critic_scheduler']:
            print(f"  {key}: {val}")
    print("=" * 40)

    # Initialize environment and Actor network
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, action_dim).to(device)
    
    if isinstance(ckpt, dict) and 'actor' in ckpt:
        actor.load_state_dict(ckpt['actor'])
    else:
        actor.load_state_dict(ckpt)
    
    actor.eval()

    # Run inference for 20 seeds (0 to 19)
    returns = []
    for seed in range(20):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        
        while not done:
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
