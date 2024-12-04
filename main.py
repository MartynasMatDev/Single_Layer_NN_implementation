import numpy as np
import matplotlib.pyplot as plt
import random
import training_data as td
import test_data as test


def tanh(x):
    return np.tanh(x)  # looks fancy, doesn't work.


def sigmoid(x):
    return 1.00 / (1.00 + np.exp(-x))


def single_neuron_output(X, W):
    weighted_sum = W[0]
    for i in range(len(X)):
        weighted_sum += X[i] * W[i + 1]
    return sigmoid(weighted_sum)


def one_layer_network(X, W):
    return [single_neuron_output(X, neuron_weights) for neuron_weights in W]


def train_neural_network(XSet, W, YSet, max_iter=50000, initial_learning_rate=0.35, tolerance=1e-7, decay_factor=0.95,
                         patience=100):
    history = [[] for _ in range(len(XSet))]
    mse_history = []
    iteration = 0
    best_mse = float('inf')
    no_improvement_count = 0
    learning_rate = initial_learning_rate

    while True:
        max_error = 0
        mse = 0
        for idx, (X, target) in enumerate(zip(XSet, YSet)):
            actual_output = one_layer_network(X, W)
            error_gradients = [
                (target[i] - actual_output[i]) * actual_output[i] * (1 - actual_output[i])
                for i in range(len(target))
            ]
            max_error = max(max_error, max(abs(g) for g in error_gradients))
            mse += np.sum((np.array(target) - np.array(actual_output)) ** 2) / len(target)
            for i, neuron_weights in enumerate(W):
                for j in range(len(neuron_weights)):
                    update = learning_rate * error_gradients[i] * (X[j - 1] if j > 0 else 1)
                    neuron_weights[j] += update

            history[idx].append(max_error)
        mse /= len(XSet)
        mse_history.append(mse)

        if mse < best_mse - tolerance:
            best_mse = mse
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            learning_rate *= decay_factor
            no_improvement_count = 0
            print(f"Learning rate adjusted to {learning_rate:.10f}")

        print(
            f"Iteration: {iteration} | Max Error: {max_error:.10f} | MSE: {mse:.10f} | Learning Rate: {learning_rate:.10f}")
        iteration += 1

        if max_error <= tolerance or iteration >= max_iter:
            break

    return W, history, mse_history


def plot_training_history(history):
    for i, errors in enumerate(history):
        plt.plot(range(1, len(errors) + 1), errors, label=f"Sample {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Max Error of gradients")
    plt.legend()
    plt.title("Training Error History")
    plt.show()


def plot_mse_history(mse_history):
    plt.plot(range(1, len(mse_history) + 1), mse_history)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs Epochs")
    plt.show()


def plot_sigmoid():
    x_values = np.linspace(-5, 5, 100)
    y_values = sigmoid(x_values)

    plt.plot(x_values, y_values, label="Sigmoid Function")
    plt.title("Sigmoid Activation Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


def test_neural_network(XSet, W, YSet):
    correct = 0
    for X, target in zip(XSet, YSet):
        output = one_layer_network(X, W)
        predicted = np.argmax(output)
        actual = np.argmax(target)
        correct += (predicted == actual)
        human_readable_output = []
        for out in output:
            rounded_value = round(float(out), 2)
            human_readable_output.append(rounded_value)
        print(f"Predicted: {predicted}, Actual: {actual}, Output: {human_readable_output}")
    print(f"Accuracy: {correct / len(XSet) * 100:.2f}%")


if __name__ == "__main__":
    input_size = 100  # 10x10 input
    num_outputs = 10  # Digits 0-9

    XSet = td.input
    NNTansSet = td.answers

    W = [[random.uniform(-0.50, 0.50) for _ in range(input_size + 1)] for _ in range(num_outputs)]
    print(f"Initial weights: {W}")

    trained_weights, training_history, mse_history = train_neural_network(XSet, W, NNTansSet)

    test_neural_network(test.XTestSet, trained_weights, test.YTestSet)
    print(f"Trained weights: {trained_weights}")

    plot_sigmoid()
    plot_training_history(training_history)
    plot_mse_history(mse_history)
