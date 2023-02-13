import numpy as np
from scipy.special import expit


def sigmoid(x):
    return expit(x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MLP:
    def __init__(self, input_shape, hidden_layer_sizes, output_shape):
        self.weights = []
        self.biases = []

        # Initialize weights for input layer
        input_weights = np.random.randn(input_shape, hidden_layer_sizes[0])
        self.weights.append(input_weights)
        input_bias = np.zeros((1, hidden_layer_sizes[0]))
        self.biases.append(input_bias)

        # Initialize weights for hidden layers
        for i in range(1, len(hidden_layer_sizes)):
            hidden_weights = np.random.randn(hidden_layer_sizes[i - 1], hidden_layer_sizes[i])
            self.weights.append(hidden_weights)
            hidden_bias = np.zeros((1, hidden_layer_sizes[i]))
            self.biases.append(hidden_bias)

        # Initialize weights for output layer
        output_weights = np.random.randn(hidden_layer_sizes[-1], output_shape)
        self.weights.append(output_weights)
        output_bias = np.zeros((1, output_shape))
        self.biases.append(output_bias)

    def forward_pass(self, X):
        # Calculate activations for input layer
        z = np.dot(X, self.weights[0]) + self.biases[0]
        a = sigmoid(z)

        activations = [a]

        # Calculate activations for hidden layers
        for i in range(1, len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            activations.append(a)

        # Calculate activations for output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = sigmoid(z)
        activations.append(a)

        return activations

