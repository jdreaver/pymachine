"""Tests for pymachine.perceptron"""

import numpy as np

from pymachine import perceptron

X = np.array([[0, 0.5], [0, 0.2], [0.5, 0.6], [0.4, 0.7], [0.9, 0.8]])
y = np.array([1, 1, -1, -1, -1])

def test_converge():
    pla = perceptron.PLA()
    pla.fit(X, y) 
    assert np.array_equal(np.sign(np.dot(pla.X, pla.weights)), y)
    
    test_weights = np.array([1, 2, 3])
    rho = perceptron.compute_rho(test_weights, pla.X)
    assert np.allclose(rho, np.array([2.5, 1.6, 3.8, 3.9, 5.2]))

    labels = perceptron.compute_labels(test_weights, pla.X)
    assert np.array_equal(labels, np.ones(5))
    assert np.array_equal(y, perceptron.compute_labels(pla.weights, pla.X))


