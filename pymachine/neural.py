"""This module implements neural networks with backpropagation."""

import numpy as np
import random


class NeuralNetwork(object):
    """
    """
    
    def __init__(self, layer_sizes, activation_fn=np.tanh):
        """
        """
        
        self.num_layers = len(layer_sizes) - 1
        self.weights = [create_weights(layer_sizes[i], layer_sizes[i + 1]-1) 
                        for i in range(self.num_layers-1)]
        self.weights.append(create_weights(layer_sizes[-2], layer_sizes[-1]))
        self.activation_fn = activation_fn
        self.activation_deriv = derivatives[activation_fn]
        
def create_weights(input_size, output_size):

    """Randomly initializes weights."""

    return np.random.rand(output_size, input_size)

@np.vectorize
def tanh_activation_deriv(x):
    return 1 - x**2
    
derivatives = {np.tanh:tanh_activation_deriv}
    
def compute_activations(x, network):
    
    """Computes activations for network given input x."""

    x = np.append(1, x)
    activations = [x]

    for i in range(network.num_layers):
        signal = np.dot(network.weights[i], activations[-1])
        if i < network.num_layers - 1:
            signal = np.append(1, signal)
        activations.append(network.activation_fn(signal))

    return activations


def compute_deltas(y, activations, network):

    """ """

    # Compute output layer
    deltas = [activations[-1] - y]
    N_l = len(activations) - 1
    for i in range(N_l):
        layer = N_l - i
        z = activations[layer - 1]
        w = network.weights[layer - 1]
        delta_next = deltas[0]
        delta_layer = network.activation_deriv(z) * np.dot(w.T, delta_next)
        deltas.insert(0, delta_layer[1:])
    
    return deltas

def network_gradient(x, y, network):
    
    """Returns derivative of E wrt to each weight."""
    
    activations = compute_activations(x, network)
    deltas = compute_deltas(y, activations, network)
    z = activations[:-1]
    d = deltas[1:]
    derivs = [np.outer(d[i], z[i]) for i in range(len(d))]

    return derivs

def modify_weights(network, gradient, eta):
    for i in range(len(network.weights)):
        network.weights[i] -= gradient[i]*eta
    return network.weights

def SGD(X, y, network, eta=0.1, tol=0.01, max_iter=500, printing=False):
    
    N = X.shape[0]
    num_epochs = 0
    while num_epochs < max_iter:
        num_epochs += 1
        if printing is True:
            print num_epochs, "/", max_iter
        order = range(N)
        random.shuffle(order)
        #old_weights = np.copy(network.weights)        
        for i in order:
            gradient = network_gradient(X[i], y[i], network)
            network.weights = modify_weights(network, gradient, eta)
        #if sum_error/N < tol:
        #    break

    return network


    
    
    
