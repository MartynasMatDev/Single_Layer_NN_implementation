# Neural Network for Digit Classification

This project implements a simple neural network for classifying handwritten digits. The network is trained using a single hidden layer, and utilizes the sigmoid activation function for learning. The training and test datasets consist of binary 10x10 grids representing digits (0-9).

---

## Project Overview

1. **Neural Network Structure**
    - Input layer: 100 neurons (10x10 grid).
    - Output layer: 10 neurons (representing digits 0-9).
    - Activation function: Sigmoid.
    - Training method: Gradient descent with error backpropagation.

2. **Features**
    - Dynamic learning rate with decay.
    - Early stopping based on error tolerance and learning rate decay patience.
    - Error metrics: Max error gradients and mean squared error (MSE).
    - Training history visualization: MSE vs Epochs and error gradients.

---

## Implementation

### Core Functions

1. **Activation Functions**
   - `sigmoid(x)`: Implements the sigmoid function.

2. **Network Operations**
   - `single_neuron_output(X, W)`: Calculates the output of a single neuron.
   - `one_layer_network(X, W)`: Calculates the output of the network layer.

3. **Training**
   - `train_neural_network(...)`: Trains the neural network using error backpropagation with support for learning rate decay and early stopping.

4. **Visualization**
   - `plot_training_history(history)`: Plots the max error gradient for all samples during training.
   - `plot_mse_history(mse_history)`: Plots the MSE evolution during training.
   - `plot_sigmoid()`: Visualizes the sigmoid activation function.

5. **Testing**
   - `test_neural_network(...)`: Tests the neural network on unseen data and computes accuracy.

6. **Dataset Utilities**
   - `render_to_console(vectors, width)`: Renders 10x10 digit grids to the console.
   - `save_as_png(vectors, dir_prefix, width)`: Saves digit grids as PNG images.

---

## Getting Started

### Prerequisites

- Python 3.7+
- `numpy`
- `matplotlib`

Install dependencies using pip:
```bash
pip install -r requirements.txt
