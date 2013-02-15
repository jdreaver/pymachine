"""This module implements linear regression."""

import numpy as np


def linear_regression(X, y):
    
    """Performs simple linear regression."""

    return np.linalg.lstsq(X, y)[0]

def weight_decay_regression(X, y, decay):
    
    """Performs linear regression with weight decay."""

    (N, dim) = X.shape
    return np.linalg.solve(np.dot(X.T, X) + decay * np.identity(dim), 
                           np.dot(X.T, y))
