
import numpy as np

import pymachine.neural as ne

network = ne.NeuralNetwork([2, 2, 1], np.tanh)
network.weights = [np.array([[1, -1, 0.5],
                             [0, -1, 0]]),
                   np.array([1, -2, 1])]

def test_tanh():
    x = np.array([1, 2, 3])
    assert np.array_equal(ne.tanh_activation_deriv(x), 1 - np.power(x, 2))

def test_backprop():
    x = np.array([1, 2])
    y = -2
    activations = ne.compute_activations(x, network)
    expected_activations = [np.array([1, 1, 2]),
                            np.array([ 0.76159416,  0.76159416, -0.76159416]),
                            -0.90925167399694251]
    for a, b in zip(activations, expected_activations):
        assert np.allclose(a, b, atol=0.001)

    deltas = ne.compute_deltas(y, activations, network.weights, 
                               network.activation_deriv)
    expected_deltas = [np.array([ 0, -0, -1.37425893]),
                       np.array([-0.45808631, 0.91617262,-0.45808631]),
                       -1.0907483260030575]
    for a, b in zip(deltas, expected_deltas):
        assert np.allclose(a, b, atol=0.001)
    


                          
    
