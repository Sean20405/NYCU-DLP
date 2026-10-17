#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

"""
Modifications:
- Gradient clipping to 0.5 (batch = 1, to stable training)

"""

import os
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, out_dim)
        self.log_std = nn.Linear(128, out_dim)
        
        initialize_uniformly(self.mu)
        initialize_uniformly(self.log_std)
        #############################
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        mu = torch.tanh(self.mu(x)) * 2.0
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        
        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        initialize_uniformly(self.out)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.eval_interval = args.eval_interval
        self.checkpoint_interval = args.checkpoint_interval
        self.checkpoint_dir = args.checkpoint_dir
        self.best_eval_reward = -float('inf')
        self.last_eval_reward = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.eval_env = gym.make("Pendulum-v1")
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        
        ############TODO#############
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state_t)
        value = self.critic(state)
        
        q_value = reward + self.gamma * next_value * mask
        value_loss = F.mse_loss(value, q_value.detach())
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        advantage = (q_value - value).detach()
        _, dist = self.actor(state)
        entropy = dist.entropy().sum(dim=-1)
        
        policy_loss = -log_prob * advantage - self.entropy_weight * entropy
        policy_loss = policy_loss.mean()
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def evaluate(self, seeds: list) -> list:
        """Evaluate the agent on given seeds."""
        self.is_test = True
        tmp_env = self.env
        self.env = self.eval_env
        
        eval_returns = []
        for seed in seeds:
            state, _ = self.env.reset(seed=seed)
            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
            eval_returns.append(score)
            
        self.env = tmp_env
        self.is_test = False
        return eval_returns

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        state, _ = self.env.reset(seed=self.seed)
        pbar = tqdm(range(1, self.num_episodes + 1))
        for ep in pbar:
            actor_losses, critic_losses, scores = [], [], []
            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                if step_count % 20 == 0:
                    wandb.log({
                        "step": step_count,
                        "actor loss": actor_loss,
                        "critic loss": critic_loss,
                        }) 
                # if episode ends
                if done:
                    scores.append(score)
                    pbar.set_postfix(step=step_count, reward=score)
                    
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score,
                        "step": step_count,
                        "actor_lr": self.actor_optimizer.param_groups[0]['lr'],
                        "critic_lr": self.critic_optimizer.param_groups[0]['lr']
                        })  
                    if ep % self.eval_interval == 0:
                        eval_seeds = [i for i in range(20)]
                        eval_returns = self.evaluate(eval_seeds)
                        mean_eval_reward = np.mean(eval_returns)
                        self.last_eval_reward = mean_eval_reward
                        wandb.log({
                            "step": step_count,
                            "eval return": mean_eval_reward,
                        })
                        
                        # Save the best model
                        if mean_eval_reward > self.best_eval_reward:
                            self.best_eval_reward = mean_eval_reward
                            best_path = os.path.join(self.checkpoint_dir, "a2c_best.pth")
                            torch.save({
                                'actor': self.actor.state_dict(),
                                'critic': self.critic.state_dict(),
                                'actor_optimizer': self.actor_optimizer.state_dict(),
                                'critic_optimizer': self.critic_optimizer.state_dict(),
                                'episode': ep,
                                'step': step_count,
                                'eval_reward': mean_eval_reward,
                            }, best_path)
                            print(f"\n[A2C] New best eval reward: {mean_eval_reward:.2f}. Saved best model to {best_path}")

                    # Save periodic checkpoint
                    if ep % self.checkpoint_interval == 0:
                        checkpoint_path = os.path.join(self.checkpoint_dir, f"a2c_checkpoint_ep{ep}.pth")
                        torch.save({
                            'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'actor_optimizer': self.actor_optimizer.state_dict(),
                            'critic_optimizer': self.critic_optimizer.state_dict(),
                            'episode': ep,
                            'step': step_count,
                            'eval_reward': self.last_eval_reward,
                        }, checkpoint_path)
                        print(f"\n[A2C] Saved periodic checkpoint to {checkpoint_path}")
        self.eval_env.close()

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="pendulum-a2c")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--eval-interval", type=int, default=10, help="evaluation interval (default: 10)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="checkpoint saving interval (default: 100)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="directory to save checkpoints")
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.exp_name, save_code=True)
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    
    agent = A2CAgent(env, args)
    agent.train()