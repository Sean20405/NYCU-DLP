import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
import glob

from wrapper import PongActionWrapper

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


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = PongActionWrapper(env) if args.act_reduce else env
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    # Find model ckpt
    ckpt_files = glob.glob(os.path.join(args.model_dir, "task3*.pt"))
    ckpt_rewards = []

    for ckpt in ckpt_files:
        print("=" * 50)
        print(f"===== Evaluating checkpoint: {ckpt} =====")

        model = DuelingDQN(4, num_actions).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        curr_output_dir = os.path.join(args.output_dir, os.path.basename(ckpt).split(".pt")[0].split("_")[1])

        os.makedirs(curr_output_dir, exist_ok=True)

        total_rewards = []

        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            state = preprocessor.reset(obs)
            done = False
            total_reward = 0
            frames = []
            frame_idx = 0

            while not done:
                frame = env.render()
                frames.append(frame)

                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(state_tensor).argmax().item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = preprocessor.step(next_obs)
                frame_idx += 1

            out_path = os.path.join(curr_output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
                for f in frames:
                    video.append_data(f)
            total_rewards.append(total_reward)
            print(f"Saved episode {ep} (seed {ep}) with total reward {total_reward} → {out_path}")

        print(f"\033[92mAverage reward over {len(total_rewards)} episodes: {np.mean(total_rewards)} for checkpoint {ckpt}\033[0m")
        ckpt_rewards.append(np.mean(total_rewards))
        print("=" * 20)

    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), "w") as f:
        for ckpt, reward in zip(ckpt_files, ckpt_rewards):
            f.write(f"{os.path.basename(ckpt)}: Average Reward = {reward}\n")

    print(f"Evaluation complete. Summary saved to {os.path.join(args.output_dir, 'evaluation_summary.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Dir to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos/dqn_pong")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--act-reduce", action="store_true", help="Whether to reduce action space to 3 (NOOP, UP, DOWN)")
    args = parser.parse_args()
    evaluate(args)
