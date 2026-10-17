#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
import random
from collections import deque
from typing import Deque, List, Tuple

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

        ############TODO#############
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, out_dim)
        self.log_std = nn.Linear(128, out_dim)
        
        init_layer_uniform(self.mu)
        init_layer_uniform(self.log_std)
        #############################

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        mu = torch.tanh(self.mu(x)) * 2.0
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
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
        
        init_layer_uniform(self.out)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    gae_returns = []
    gae = 0
    all_values = values + [next_value]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * all_values[t+1] * masks[t] - all_values[t]
        gae = delta + gamma * tau * masks[t] * gae
        gae_returns.insert(0, gae + all_values[t])
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.args = args
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.eval_interval = args.eval_interval
        self.checkpoint_interval = args.checkpoint_interval
        self.checkpoint_dir = args.checkpoint_dir
        self.max_grad_norm = args.max_grad_norm
        self.best_eval_reward = -float('inf')
        self.last_eval_reward = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.eval_env = gym.make("Pendulum-v1")
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

        # History lists to log training stats
        self.history = {
            'episodes': [],
            'steps': [],
            'returns': [],
            'eval_steps': [],
            'eval_returns': [],
            'actor_losses': [],
            'critic_losses': []
        }

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            actor_loss = -torch.min(surr1, surr2) - self.entropy_weight * entropy
            actor_loss = actor_loss.mean()
            #############################

            # critic_loss
            ############TODO#############
            value = self.critic(state)
            critic_loss = F.mse_loss(value, return_)
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def evaluate(self, seeds: list) -> list:
        """Evaluate the agent on given seeds."""
        self.is_test = True
        tmp_env = self.env
        self.env = self.eval_env
        
        eval_returns = []
        for seed in seeds:
            state, _ = self.env.reset(seed=seed)
            state = np.expand_dims(state, axis=0)
            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward[0][0]
                if isinstance(done, np.ndarray):
                    done = done[0][0]
            eval_returns.append(score)
            
        self.env = tmp_env
        self.is_test = False
        return eval_returns

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        pbar = tqdm(range(1, self.num_episodes + 1))
        for ep in pbar:
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    pbar.set_postfix(step=self.total_step, reward=score)
                    wandb.log({
                        "episode": episode_count,
                        "return": score,
                        "step": self.total_step
                    })
                    self.history['episodes'].append(episode_count)
                    self.history['steps'].append(self.total_step)
                    self.history['returns'].append(score)
                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            self.history['actor_losses'].append(actor_loss)
            self.history['critic_losses'].append(critic_loss)

            wandb.log({
                "step": self.total_step,
                "actor loss": actor_loss,
                "critic loss": critic_loss,
                "actor_lr": self.actor_optimizer.param_groups[0]['lr'],
                "critic_lr": self.critic_optimizer.param_groups[0]['lr']
            })

            if ep % self.eval_interval == 0:
                eval_seeds = [i for i in range(20)]
                eval_returns = self.evaluate(eval_seeds)
                mean_eval_reward = np.mean(eval_returns)
                self.last_eval_reward = mean_eval_reward
                wandb.log({
                    "step": self.total_step,
                    "eval return": mean_eval_reward,
                })
                self.history['eval_steps'].append(self.total_step)
                self.history['eval_returns'].append(mean_eval_reward)
                
                # Save the best model
                if mean_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = mean_eval_reward
                    best_path = os.path.join(self.checkpoint_dir, "ppo_best.pth")
                    torch.save({
                        'actor': self.actor.state_dict(),
                        'critic': self.critic.state_dict(),
                        'actor_optimizer': self.actor_optimizer.state_dict(),
                        'critic_optimizer': self.critic_optimizer.state_dict(),
                        'episode': ep,
                        'step': self.total_step,
                        'eval_reward': mean_eval_reward,
                    }, best_path)
                    print(f"\n[PPO] New best eval reward: {mean_eval_reward:.2f}. Saved best model to {best_path}")

            # Save periodic checkpoint
            if ep % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"ppo_checkpoint_ep{ep}.pth")
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'episode': ep,
                    'step': self.total_step,
                    'eval_reward': self.last_eval_reward,
                }, checkpoint_path)
                print(f"\n[PPO] Saved periodic checkpoint to {checkpoint_path}")

        # Dump training history to JSON
        os.makedirs(self.args.log_dir, exist_ok=True)
        import json
        log_path = os.path.join(self.args.log_dir, f"{self.args.exp_name}.json")
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"\n[PPO] Saved training history to {log_path}")

        # termination
        self.eval_env.close()
        self.env.close()

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
    parser.add_argument("--exp-name", type=str, default="pendulum-ppo")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5, help="evaluation interval (default: 5)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="checkpoint saving interval (default: 10)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max norm of gradients (default: 0.5)")
    parser.add_argument("--log-dir", type=str, default="results", help="directory to save JSON training results")
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.exp_name, save_code=True)
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    
    agent = PPOAgent(env, args)
    agent.train()