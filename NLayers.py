import numpy as np
from scipy.special import expit


def create_mlp(input_shape, hidden_layer_sizes, output_shape, weights):
    activations = [input_shape] + hidden_layer_sizes + [output_shape]
    n_layers = len(activations) - 1

    for i in range(n_layers):
        layer_weights = weights[i]
        layer_input = activations[i]
        layer_output = activations[i + 1]

        z = np.dot(layer_input, layer_weights)
        a = expit(z)

        activations[i + 1] = a

    return activations[-1]
