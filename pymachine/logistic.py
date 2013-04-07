"""This module implements the functions for logistic regression."""

import numpy as np
import random

def logistic_gradient(x, y, weights):
    return -y*x/(1 + np.exp(y * np.dot(x, weights)))

def logistic_error(X, y, weights):
    return np.mean(np.log(1 + np.exp(-y * np.dot(X, weights))))

def cross_entropy_error(X, y, weights):
    return np.mean(np.log(1 + np.exp(-y * np.dot(weights, X.T))))


def logistic_gradient_descent(X, y, tol, eta, max_iter):

    """Stochastic gradient descent for logistic regression.

    This function performs logistic regression using stochastic
    gradient descent.

    Args:
        X: numpy ndarray of shape (num_points, dim)
        y: labels for X in {-1, 1}
        tol: tolerance for halting condition using norm of change in weights
        eta: learning rate
        max_iter: maximum number of epochs

    Returns:
        weights: the final weights of linear separator
        num_epochs: number of epochs before termination
    """
 
    (N, dim) = X.shape
    X = np.hstack([np.ones((N, 1)), X])
    weights = np.zeros(dim + 1)

    num_epochs = 0
    while num_epochs < max_iter:
        num_epochs += 1
        order = list(range(N))
        random.shuffle(order)
        old_weights = np.copy(weights)
        for i in order:
            weights -= eta*logistic_gradient(X[i], y[i], weights)
        #print weights, old_weights, np.linalg.norm(old_weights - weights)
        if np.linalg.norm(old_weights - weights) < tol:
            break

    return (weights, num_epochs)

        
    
    
    

