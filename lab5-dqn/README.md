# Lab 5: Value-Based Reinforcement Learning

DQN on CartPole-v1 and on Pong (ALE/Pong-v5). The full agent for Pong adds Double DQN, Prioritized Experience Replay (PER), multi-step returns, a dueling network, and action reduction.

| Task | Environment | Agent | Script |
|---|---|---|---|
| 1 | CartPole-v1 | Baseline DQN | `src/dqn_task1.py` |
| 2 | Pong | Baseline DQN | `src/dqn_task2.py` |
| 3 | Pong | Full DQN with all techniques | `src/dqn_task3.py` |

## Setup

```bash
conda create -y -n dlp_lab5 python=3.12
conda activate dlp_lab5
pip install -r src/requirements.txt
```

## Train

```bash
python src/dqn_task1.py
python src/dqn_task2.py
python src/dqn_task3.py --dueling --double-q --per --multi-step 3 --action-reduction
```

`src/exp.sh` runs the ablation: the full agent and the agent with one technique removed. `src/wrapper.py` is the Atari frame preprocessing.

## Test

You must train first, because the weights are not in this repository. Each test script loads a checkpoint and records evaluation episodes.

```bash
python src/test_model_task1.py --model-path task1.pt --output-dir eval_videos/task1
python src/test_model_task2.py --model-path task2.pt --output-dir eval_videos/task2
python src/test_model_task3.py --model-dir . --output-dir eval_videos/task3
```

## Results

The full agent learns faster than the baseline. The baseline needed about 800k steps to reach a positive score. The full agent needed about 350k to 400k steps. It reached 19 or higher at 1.5M steps, while the baseline needed almost 4M steps to reach a stable plateau.

Average evaluation reward on Pong at fixed training steps:

| Method | 1M | 1.5M | 2M | 2.5M |
|---|---|---|---|---|
| Baseline DQN | 16.45 | 16.1 | 16.8 | 16.45 |
| Full agent | 14.95 | 18.2 | 19.4 | 18.85 |
| Without Double DQN | 7 | 5.67 | 15 | 17.33 |
| Without PER | 7.67 | 12 | 16.33 | 13 |
| Without multi-step | 14.33 | 18.33 | 17.33 | 18.67 |
| Without dueling | 16.67 | 18.33 | 17 | 19 |

PER and Double DQN matter most. The best single run reached 19.95 at 2.5M steps, and the results vary much between runs. The raw evaluation summaries are in `results/eval/`. The full analysis is in [report/report.pdf](report/report.pdf).
