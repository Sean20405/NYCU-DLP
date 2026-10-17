# Lab 1: Backpropagation

A three-layer MLP with sigmoid activations. It uses only NumPy, with the forward and backward passes written by hand. The network trains on a linearly separable dataset and on the XOR dataset.

## Setup

You need Python 3 and NumPy. You also need Matplotlib to save the plots.

## Run

```bash
python src/main.py --data linear
python src/main.py --data XOR
```

The other options are `--lr`, `--hidden_size`, `--activation` (`sigmoid`, `relu`, `tanh`, `none`), and `--seed`. The defaults are learning rate 5e-3, 16 hidden units, and 10,000 epochs.

`src/run_experiments.py` runs the learning rate, hidden size, and activation experiments from the report. Run it from inside `src/`, because it imports `main.py`. It writes the plots to `logs/`.

## Results

Both datasets reach 100% test accuracy with the default configuration. The plots in `results/` show the learning curves and the predictions for each experiment.

- With the XOR dataset, learning rates 0.1 and 0.01 converge to 100% accuracy. Rates 0.001 and 0.0001 are too slow for 10,000 epochs.
- With the linear dataset, learning rate 0.1 fails and stays at 50% accuracy, because the sigmoid saturates.

The full analysis is in [report/report.pdf](report/report.pdf).
