import os
import numpy as np
import matplotlib.pyplot as plt

# Import necessary classes and functions from the existing files
from main import SimpleNet, MSELoss
from data import generate_linear, generate_XOR_easy

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def train_and_eval(lr, hidden_size, activation, x_train, y_train, x_test, y_test):
    np.random.seed(0)
    input_size = 2
    output_size = 1
    epochs = 10000

    net = SimpleNet(input_size, hidden_size, output_size, lr, activation)
    loss_fn = MSELoss()
    
    losses = []

    for epoch in range(epochs):
        y_pred = net.forward(x_train)
        loss = loss_fn.forward(y_pred, y_train)
        losses.append(loss)
        loss_grad = loss_fn.backward(y_pred, y_train)
        net.backward(loss_grad)
        net.update()

    y_pred = np.round(net.forward(x_test), decimals=0)
    acc = 1 - np.mean(abs(y_test - y_pred))
    
    return losses, y_pred, acc

def plot_combined_learning_curve(exp_name, data_name, all_losses, labels):
    plt.figure(figsize=(10, 6))
    for losses, label in zip(all_losses, labels):
        plt.plot(range(len(losses)), losses, label=label)
    plt.title(f"Learning Curve - {exp_name} ({data_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, f"learning_curve_{exp_name}_{data_name}.png"))
    plt.close()

def plot_combined_results(exp_name, data_name, x_test, y_test, all_preds, all_accs, labels):
    n_plots = 1 + len(all_preds)
    plt.figure(figsize=(4 * n_plots, 4))
    
    # Ground truth
    plt.subplot(1, n_plots, 1)
    plt.title('Ground truth', fontsize=14)
    for i in range(x_test.shape[0]):
        c = 'r' if y_test[i] == 0 else 'b'
        plt.scatter(x_test[i][0], x_test[i][1], c=c)

    # Predictions
    for idx, (preds, acc, label) in enumerate(zip(all_preds, all_accs, labels)):
        plt.subplot(1, n_plots, idx + 2)
        plt.title(f'{label}\nacc: {int(acc * 100)}%', fontsize=14)
        for i in range(x_test.shape[0]):
            c = 'r' if preds[i] == 0 else 'b'
            plt.scatter(x_test[i][0], x_test[i][1], c=c)
            
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"result_{exp_name}_{data_name}.png"))
    plt.close()

def run_experiment_set(exp_name, data_name, base_lr, base_hs, base_act, varying_param, param_values):
    print(f"Running {exp_name} on {data_name}...")
    
    if data_name == "linear":
        gen_data_fn = generate_linear
    elif data_name == "XOR":
        gen_data_fn = lambda _: generate_XOR_easy()

    x_train, y_train = gen_data_fn(1000)
    x_test, y_test = gen_data_fn(100)
    
    all_losses = []
    all_preds = []
    all_accs = []
    labels = []

    for val in param_values:
        lr = val if varying_param == "lr" else base_lr
        hs = val if varying_param == "hs" else base_hs
        act = val if varying_param == "act" else base_act
        
        label = f"{varying_param}={val}"
        labels.append(label)
        
        losses, preds, acc = train_and_eval(lr, hs, act, x_train, y_train, x_test, y_test)
        all_losses.append(losses)
        all_preds.append(preds)
        all_accs.append(acc)

    plot_combined_learning_curve(exp_name, data_name, all_losses, labels)
    plot_combined_results(exp_name, data_name, x_test, y_test, all_preds, all_accs, labels)

# Run experiments for learning rates
for data in ["linear", "XOR"]:
    run_experiment_set("exp_lr", data, base_lr=None, base_hs=16, base_act="sigmoid", varying_param="lr", param_values=[0.1, 0.01, 0.001, 0.0001])

# Run experiments for hidden units
for data in ["linear", "XOR"]:
    run_experiment_set("exp_hs", data, base_lr=5e-3, base_hs=None, base_act="sigmoid", varying_param="hs", param_values=[4, 16, 64, 256])

# Run experiments w/ and w/o activation functions
for data in ["linear", "XOR"]:
    run_experiment_set("exp_have_act", data, base_lr=5e-3, base_hs=16, base_act=None, varying_param="act", param_values=["sigmoid", "none"])

# Run experiments for different activation functions
for data in ["linear"]:
    run_experiment_set("exp_act", data, base_lr=1e-3, base_hs=16, base_act=None, varying_param="act", param_values=["sigmoid", "relu", "tanh"])

print("Combined experiments plotted successively.")
