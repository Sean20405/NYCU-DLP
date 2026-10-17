import argparse
import numpy as np
import matplotlib.pyplot as plt
from data import generate_linear, generate_XOR_easy, show_result


class Linear:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x (np.ndarray): Tensor of shape (batch_size, input_size).
        Returns:
            np.ndarray: Tensor of shape (batch_size, output_size).
        """
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Args:
            grad (np.ndarray):  Tensor of shape (batch_size, output_size).
        Returns:
            np.ndarray: Tensor of shape (batch_size, input_size).
        """
        return np.dot(grad, self.weights.T)

    def update(self, lr: float, input: np.ndarray, grad: np.ndarray):
        self.weights -= lr * np.dot(input.T, grad)
        self.bias -= lr * np.sum(grad, axis=0)


# --------------------------- Activations --------------------------
class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x) * (1 - self.forward(x))


class Identity:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class ReLU:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Tanh:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class SimpleNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float = 5e-3,
        activation: str = "sigmoid"
    ):
        self.input_layer = Linear(input_size, hidden_size)
        self.hidden_layer = Linear(hidden_size, hidden_size)
        self.output_layer = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        elif activation == "none":
            self.activation = Identity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.z1 = self.input_layer.forward(x)
        self.hidden1 = self.activation.forward(self.z1)
        self.z2 = self.hidden_layer.forward(self.hidden1)
        self.hidden2 = self.activation.forward(self.z2)
        self.z3 = self.output_layer.forward(self.hidden2)
        self.output = self.sigmoid.forward(self.z3)
        return self.output

    def backward(self, grad_loss: np.ndarray):
        self.grad_output = grad_loss * self.sigmoid.backward(self.z3)
        self.grad_hidden2 = self.output_layer.backward(self.grad_output) * self.activation.backward(self.z2)
        self.grad_hidden1 = self.hidden_layer.backward(self.grad_hidden2) * self.activation.backward(self.z1)

    def update(self):
        self.output_layer.update(self.lr, self.hidden2, self.grad_output)
        self.hidden_layer.update(self.lr, self.hidden1, self.grad_hidden2)
        self.input_layer.update(self.lr, self.input, self.grad_hidden1)


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
        
class MSELoss:
    def __init__(self):
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple neural network.")
    parser.add_argument("--data", type=str, choices=["linear", "XOR"], default="linear", help="Dataset to use: linear or XOR")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=16, help="Number of hidden units")
    parser.add_argument("--activation", type=str, choices=["sigmoid", "relu", "tanh", "none"], default="sigmoid", help="Activation function")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    input_size = 2
    output_size = 1
    hidden_size = args.hidden_size
    n_sample = 1000
    epochs = 10000
    lr = args.lr

    log_interval = 500

    net = SimpleNet(input_size, hidden_size, output_size, lr, args.activation)
    loss_fn = MSELoss()
    
    if args.data == "linear":
        gen_data_fn = generate_linear
    elif args.data == "XOR":
        gen_data_fn = lambda _: generate_XOR_easy()

    # Train
    print("===== Training =====")
    x_train, y_train = gen_data_fn(n_sample)

    losses = []

    for epoch in range(epochs):
        y_pred = net.forward(x_train)
        loss = loss_fn.forward(y_pred, y_train)
        losses.append(loss)
        loss_grad = loss_fn.backward(y_pred, y_train)
        net.backward(loss_grad)
        net.update()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    plt.figure()
    plt.plot(range(epochs), losses)
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"learning_curve_{args.data}.png")
    plt.close()

    # Test
    print("===== Testing =====")
    x_test, y_test = gen_data_fn(100)
    y_pred = np.round(net.forward(x_test), decimals=0)
    acc = 1 - np.mean(abs(y_test - y_pred))

    print(f"acc: {int(acc * 100)}%")

    show_result(x_test, y_test, y_pred)
