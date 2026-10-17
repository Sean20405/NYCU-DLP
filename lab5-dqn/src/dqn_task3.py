# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh
#
# Improvements over baseline:
# 1. Dueling DQN architecture  (--dueling)
# 2. SumTree-based PER  → O(log N) sampling instead of O(N)
# 3. PER fixes: alpha=0.5, proper beta annealing, correct priority storage
# 4. Gradient clipping (max_norm=10)
# 5. Soft target network update option  (--soft-update)
# 6. Increased default batch size 64→256
# 7. Burn-in fill uses max_priority correctly for new transitions
# 8. Action reduction: Pong 6 actions → 3 (NOOP/UP/DOWN), doubles explore efficiency

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)

EPS = 1e-6
SAVED_ENV_STEP = [600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    """Standard DQN (kept for backward-compatibility / ablation)."""
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class DuelingDQN(nn.Module):
    """
    Dueling DQN (Wang et al., 2016) https://arxiv.org/abs/1511.06581
    Q(s,a) = V(s) + A(s,a) - mean_a[ A(s,a) ]

    Why it helps for Pong:
    - Separates learning *how good this state is* from *which action is best*.
    - In Pong many frames are neutral (ball far away); the value stream can
      learn that quickly without noisy action signals.
    """
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Shared feature → two separate heads
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        feat = self.conv(x / 255.0)
        V = self.value_stream(feat)                          # (B, 1)
        A = self.advantage_stream(feat)                      # (B, |A|)
        return V + (A - A.mean(dim=1, keepdim=True))        # (B, |A|)


# ─────────────────────────────────────────────────────────────────────────────
# Action Reduction Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PongActionWrapper(gym.ActionWrapper):
    """
    Pong has 6 actions but only 3 matter:
        0 NOOP  →  0
        2 RIGHT (paddle up)   →  1
        3 LEFT  (paddle down) →  2

    Action IDs 1 (FIRE), 4 (RIGHTFIRE), 5 (LEFTFIRE) are functionally
    identical to NOOP / RIGHT / LEFT in Pong.

    Reducing from 6 → 3 means:
      - ε-greedy picks a useful action 2× more often
      - Q-network doesn't waste capacity on duplicate outputs
      - Policy converges faster
    """
    # Mapping: reduced index → original ALE action
    _ACTIONS = [0, 2, 3]   # NOOP, UP, DOWN

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self._ACTIONS))

    def action(self, act):
        return self._ACTIONS[act]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(self.preprocess(obs))
        return np.stack(self.frames, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# SumTree — O(log N) priority operations
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary segment tree.
    - tree[1]          = sum of all priorities (root)
    - tree[capacity]   .. tree[2*capacity-1]  = leaf priorities
    - data[i]          = experience at leaf i

    Problem with original code:
      np.random.choice(500_000, p=probs) is O(N) per sample call.
      With batch_size=256 and 500k buffer that's ~128M ops per update.
      SumTree samples in O(log N) ≈ 19 ops per sample → ~50× faster.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)   # 1-indexed
        self.data = [None] * capacity
        self.n_entries = 0
        self.ptr = 0
        self._max_p = 1.0          # tracks maximum leaf priority (in α-space)

    # ── internal helpers ──────────────────────────────────────────────────────
    def _propagate(self, leaf_idx: int, delta: float):
        idx = leaf_idx
        while idx > 1:
            idx >>= 1
            self.tree[idx] += delta

    def _retrieve(self, value: float) -> int:
        """Walk tree to find leaf whose cumulative sum >= value."""
        idx = 1
        while idx < self.capacity:
            left = idx << 1
            if self.tree[left] >= value:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity          # data index

    # ── public API ────────────────────────────────────────────────────────────
    def add(self, priority: float, data):
        """Add new experience with given priority (already in α-space)."""
        leaf_idx = self.ptr + self.capacity
        self.data[self.ptr] = data
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)
        if priority > self._max_p:
            self._max_p = priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, data_idx: int, priority: float):
        """Update priority for an existing experience."""
        leaf_idx = data_idx + self.capacity
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)
        if priority > self._max_p:
            self._max_p = priority

    def sample(self, batch_size: int):
        """
        Stratified sampling: divide [0, total] into batch_size segments and
        sample one value uniformly from each segment → lower variance than
        pure uniform sampling over [0, total].
        """
        total = self.tree[1]
        segment = total / batch_size
        data_indices, priorities = [], []
        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            v = np.clip(v, EPS, total - EPS)
            data_idx = self._retrieve(v)
            data_indices.append(data_idx)
            priorities.append(self.tree[data_idx + self.capacity])
        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / total
        return data_indices, probs

    @property
    def total(self) -> float:
        return float(self.tree[1])

    @property
    def max_priority(self) -> float:
        return self._max_p


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer (SumTree backend)
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Fixes vs original implementation:
    1. SumTree backend → O(log N) instead of O(N) per update/sample.
    2. alpha=0.5 instead of 0.6.
       Higher alpha → more aggressive prioritization → IS weights damp
       high-error updates MORE → slows early learning.  0.5 is a better
       balance between prioritization and stability.
    3. Priorities stored in α-space (pre-exponentiated).
       New transitions get max_priority immediately so they're always
       sampled at least once.
    4. Beta annealed from beta_start→1.0 linearly over total_env_steps
       (called externally via update_beta(env_step)).
    """
    def __init__(self, capacity, alpha=0.5, beta_start=0.4,
                 total_steps=2_500_000, multi_step=1, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.total_steps = total_steps
        self.tree = SumTree(capacity)

    # ── public API ────────────────────────────────────────────────────────────
    def add(self, transition):
        """New transitions get max priority → guaranteed to be sampled once."""
        priority = self.tree.max_priority       # already in α-space
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        data_indices, probs = self.tree.sample(batch_size)
        N = self.tree.n_entries
        # IS weights  w_i = (1 / (N * P(i)))^β,  normalised by max weight
        weights = (N * probs) ** (-self.beta)
        weights /= weights.max()
        samples = [self.tree.data[idx] for idx in data_indices]
        return samples, data_indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        """errors: numpy array of |TD errors|"""
        for idx, err in zip(indices, errors):
            priority = (float(abs(err)) + EPS) ** self.alpha
            self.tree.update(idx, priority)

    def update_beta(self, env_step: int):
        """Call this every env step to linearly anneal beta 0.4 → 1.0."""
        frac = min(1.0, env_step / self.total_steps)
        self.beta = self.beta_start + (1.0 - self.beta_start) * frac

    def __len__(self):
        return self.tree.n_entries


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        base_env      = gym.make(env_name, render_mode="rgb_array")
        base_test_env = gym.make(env_name, render_mode="rgb_array")

        if args.action_reduction:
            self.env      = PongActionWrapper(base_env)
            self.test_env = PongActionWrapper(base_test_env)
        else:
            self.env      = base_env
            self.test_env = base_test_env
        self.preprocessor = AtariPreprocessor()
        self.num_spaces   = self.preprocessor.frame_stack
        self.num_actions  = self.env.action_space.n
        self.args         = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # ── Replay buffer ────────────────────────────────────────────────────
        if args.per:
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size,
                alpha=args.per_alpha,
                beta_start=args.per_beta,
                total_steps=args.total_env_steps,
                multi_step=args.multi_step,
                gamma=args.discount_factor,
            )
        else:
            self.memory = deque(maxlen=args.memory_size)

        # ── Networks ─────────────────────────────────────────────────────────
        NetClass = DuelingDQN if args.dueling else DQN
        self.q_net     = NetClass(self.num_spaces, self.num_actions).to(self.device)
        self.target_net = NetClass(self.num_spaces, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        # ── Hyper-parameters ─────────────────────────────────────────────────
        self.batch_size             = args.batch_size
        self.gamma                  = args.discount_factor
        self.epsilon                = args.epsilon_start
        self.epsilon_decay          = args.epsilon_decay
        self.epsilon_min            = args.epsilon_min
        self.env_count              = 0
        self.train_count            = 0
        self.best_reward            = -21
        self.max_episode_steps      = args.max_episode_steps
        self.replay_start_size      = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step         = args.train_per_step
        self.save_dir               = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ── Action selection ─────────────────────────────────────────────────────
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_tensor).argmax().item()

    # ── Main training loop ───────────────────────────────────────────────────
    def run(self, episodes=1000):
        eval_rewards = []
        multi_step_buffer = deque(maxlen=self.args.multi_step)

        for ep in range(episodes):
            obs, _ = self.env.reset()
            state  = self.preprocessor.reset(obs)
            done   = False
            total_reward = 0
            step_count   = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done       = terminated or truncated
                next_state = self.preprocessor.step(next_obs)

                # ── Store transition ─────────────────────────────────────────
                if self.args.multi_step > 1:
                    multi_step_buffer.append({
                        'state': state, 'action': action, 'reward': reward,
                        'next_state': next_state, 'done': done
                    })
                    if len(multi_step_buffer) == self.args.multi_step or done:
                        n_return, gamma_p = 0.0, 1.0
                        for t in multi_step_buffer:
                            n_return += t['reward'] * gamma_p
                            gamma_p  *= self.gamma
                            if t['done']:
                                break
                        first = multi_step_buffer[0]
                        last  = multi_step_buffer[-1]
                        transition = (first['state'], first['action'],
                                      n_return, last['next_state'], last['done'])
                        self._store(transition)
                        if done:
                            multi_step_buffer.clear()
                else:
                    self._store((state, action, reward, next_state, done))

                # ── Gradient updates ─────────────────────────────────────────
                for _ in range(self.train_per_step):
                    self.train()

                state        = next_state
                total_reward += reward
                self.env_count += 1
                step_count   += 1

                # Anneal beta every env step
                if self.args.per:
                    self.memory.update_beta(self.env_count)

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep:{ep} SC:{self.env_count} "
                          f"UC:{self.train_count} Eps:{self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep, "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })

            print(f"[Eval] Ep:{ep} Reward:{total_reward} "
                  f"SC:{self.env_count} UC:{self.train_count} Eps:{self.epsilon:.4f}")
            wandb.log({
                "Episode": ep, "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })

            if ep % 100 == 0:
                ckpt = os.path.join(self.save_dir, f"model_ep{ep}_step{self.env_count}.pt")
                torch.save(self.q_net.state_dict(), ckpt)
                print(f"Saved checkpoint → {ckpt}")

            if SAVED_ENV_STEP and self.env_count > min(SAVED_ENV_STEP):
                ckpt = os.path.join(self.save_dir, f"model_step{self.env_count}.pt")
                torch.save(self.q_net.state_dict(), ckpt)
                SAVED_ENV_STEP.remove(min(SAVED_ENV_STEP))

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                eval_rewards.append(eval_reward)
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), best_path)
                    print(f"New best → {best_path}  reward={eval_reward:.2f}")
                print(f"[TrueEval] Ep:{ep} Eval:{eval_reward:.2f} SC:{self.env_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                })

        ckpt = os.path.join(self.save_dir, f"model_step{self.env_count}.pt")
        torch.save(self.q_net.state_dict(), ckpt)
        print(f"Final model → {ckpt}")

    def _store(self, transition):
        if self.args.per:
            self.memory.add(transition)
        else:
            self.memory.append(transition)

    # ── Evaluation ───────────────────────────────────────────────────────────
    def evaluate(self):
        total = 0
        for ep in range(3):
            obs, _ = self.test_env.reset(seed=ep)
            state  = self.preprocessor.reset(obs)
            done   = False
            while not done:
                st = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(st).argmax().item()
                next_obs, r, term, trunc, _ = self.test_env.step(action)
                done   = term or trunc
                total += r
                state  = self.preprocessor.step(next_obs)
        return total / 3

    # ── Single training step ─────────────────────────────────────────────────
    def train(self):
        # ── Wait until buffer is ready ───────────────────────────────────────
        buf_len = len(self.memory) if self.args.per else len(self.memory)
        if buf_len < self.replay_start_size:
            return

        # ── Epsilon decay (per gradient step) ───────────────────────────────
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # ── Sample ───────────────────────────────────────────────────────────
        if self.args.per:
            batch, indices, weights = self.memory.sample(self.batch_size)
            weights_t = torch.from_numpy(weights).float().to(self.device)
        else:
            batch = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.from_numpy(np.array(states,      dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions     = torch.tensor(actions,  dtype=torch.int64  ).to(self.device)
        rewards_t   = torch.tensor(rewards,  dtype=torch.float32).to(self.device)
        dones_t     = torch.tensor(dones,    dtype=torch.float32).to(self.device)

        # ── Current Q values ─────────────────────────────────────────────────
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Target Q values (Double DQN) ─────────────────────────────────────
        discount = self.gamma ** self.args.multi_step
        with torch.no_grad():
            if self.args.double_q:
                # Action selection by online net, value evaluation by target net
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards_t + discount * next_q * (1 - dones_t)

        # ── Loss ─────────────────────────────────────────────────────────────
        td_errors = (q_values - target_q).abs().detach()

        if self.args.per:
            # Element-wise Huber loss, re-weighted by IS weights
            loss = nn.SmoothL1Loss(reduction='none')(q_values, target_q)
            loss = (loss * weights_t).mean()
            self.memory.update_priorities(indices, td_errors.cpu().numpy())
        else:
            loss = nn.SmoothL1Loss()(q_values, target_q)

        # ── Gradient update with clipping ────────────────────────────────────
        # FIX: without clipping, large TD errors (esp. with n-step returns)
        #      produce exploding gradients that destabilise early training.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Target network update ─────────────────────────────────────────────
        if self.args.soft_update:
            # Polyak averaging — smoother than hard copy every N steps
            tau = self.args.soft_update_tau
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        else:
            if self.train_count % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] loss:{loss.item():.4f} "
                  f"Q_mean:{q_values.mean().item():.3f} "
                  f"Q_std:{q_values.std().item():.3f} "
                  f"beta:{self.memory.beta:.3f}" if self.args.per else "")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name",   type=str,   default="enhanced_dqn_pong")
    parser.add_argument("--save-dir",   type=str,   default="./results")
    parser.add_argument("--episodes",   type=int,   default=1_200)

    # Core DQN hyper-parameters
    parser.add_argument("--batch-size",               type=int,   default=256)       # ↑ from 64
    parser.add_argument("--memory-size",              type=int,   default=500_000)
    parser.add_argument("--lr",                       type=float, default=6.25e-5)
    parser.add_argument("--discount-factor",          type=float, default=0.99)
    parser.add_argument("--epsilon-start",            type=float, default=1.0)
    parser.add_argument("--epsilon-decay",            type=float, default=0.99997)
    parser.add_argument("--epsilon-min",              type=float, default=0.01)
    parser.add_argument("--target-update-frequency",  type=int,   default=2_000)
    parser.add_argument("--replay-start-size",        type=int,   default=20_000)
    parser.add_argument("--max-episode-steps",        type=int,   default=10_000)
    parser.add_argument("--train-per-step",           type=int,   default=1)
    parser.add_argument("--total-env-steps",          type=int,   default=2_500_000)

    # Algorithmic flags
    parser.add_argument("--double-q",    action="store_true", help="Double DQN")
    parser.add_argument("--dueling",     action="store_true", help="Dueling DQN architecture")
    parser.add_argument("--per",         action="store_true", help="Prioritized Experience Replay")
    parser.add_argument("--multi-step",  type=int, default=3, help="N-step returns")
    parser.add_argument("--action-reduction", action="store_true",
                        help="Reduce Pong actions from 6 to 3 (NOOP/UP/DOWN)")

    # PER hyper-parameters
    parser.add_argument("--per-alpha",   type=float, default=0.5,  # ↓ from 0.6
                        help="PER priority exponent (0=uniform, 1=fully prioritised)")
    parser.add_argument("--per-beta",    type=float, default=0.4,
                        help="PER IS-weight starting exponent (annealed to 1.0)")

    # Soft target update
    parser.add_argument("--soft-update",     action="store_true",
                        help="Polyak target update instead of hard copy")
    parser.add_argument("--soft-update-tau", type=float, default=0.005,
                        help="Polyak averaging coefficient τ")

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.exp_name)

    print("Config:")
    print(f"  Dueling DQN  : {args.dueling}")
    print(f"  Double  DQN  : {args.double_q}")
    print(f"  PER          : {args.per}  (α={args.per_alpha}, β₀={args.per_beta})")
    print(f"  Multi-step   : {args.multi_step}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Soft update  : {args.soft_update}  (τ={args.soft_update_tau})")
    print(f"  Action reduction  : {args.action_reduction}  (6 → 3 actions)\n")


    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.exp_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(episodes=args.episodes)