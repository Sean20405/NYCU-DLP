import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def generate_linear(n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy() -> Tuple[np.ndarray, np.ndarray]:
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x: np.ndarray, y: np.ndarray, pred_y: np.ndarray):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)

    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], c='r')
        else:
            plt.scatter(x[i][0], x[i][1], c='b')
    
    plt.subplot(1, 2, 2)
    plt.title('Prediction result', fontsize=18)

    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.scatter(x[i][0], x[i][1], c='r')
        else:
            plt.scatter(x[i][0], x[i][1], c='b')
    
    plt.savefig("result.png")