"""
This module implements the perceptron learning algorithm. 
"""

import numpy as np
from pandas import DataFrame

from pymachine import datagen

class PLA():

    """
    This class holds the initial parameters and methods of the
    perceptron learning algorithm. 
    """

    def __init__(self, dimension=2, damping=None, bounds=None):
        if bounds is None:
            self.bounds = datagen.unit_bounds(dimension)
        else:
            self.bounds = bounds
        self.damping = damping

    def fit(self, feature_matrix, labels, maxiter=1000):
        self.data = features_to_pla(feature_matrix)
        self.X = np.array(self.data)
        self.y = labels
        (self.weights, self.num_iters) = run_pla(self.X, self.y, maxiter=maxiter)


def compute_rho(weights, X):
    return np.dot(X, weights)

def compute_labels(weights, X):
    return np.sign(compute_rho(weights, X))

def update_weights(weights, X, y, damping=None):

    """Returns next iteration of weights for PLA.

    Computes the next iteration of weights for the perceptron learning
    algorithm. 

    """

    computed_labels = compute_labels(weights, X)

    # Select a random misclassified point
    misclassified = np.where(computed_labels != y)[0]
    if misclassified.size == 0:
        return weights
    index = misclassified[np.random.randint(len(misclassified))]

    if damping:
        rho = compute_rho(weights, X[index])
        new_weights = weights + damping*np.dot((y[index]-rho), X[index])
    else:
        new_weights = weights + y[index]*X[index]
    return new_weights
        

def run_pla(X, y, weights=None, maxiter=1000):

    """Runs PLA on set of labeled features."""

    n = 0
    if weights == None:
        weights = np.zeros(X.shape[1])
    old_weights = np.ones(len(weights))

    while not np.array_equal(weights, old_weights) and n < maxiter:
        old_weights = weights
        weights = update_weights(weights, X, y)
        n += 1
    return (weights, n)


def features_to_pla(features):

    """Transforms feature matrix to PLA form with bias term.

    Takes a given feature matrix and converts it to a pandas DataFrame
    with and extra bias term.

    Args:
        features: matrix of features

    Returns:
        DataFrame of features with bias term
    """
    
    (N, dimension) = features.shape
    if isinstance(features, DataFrame):
        labels = np.append('bias', features.columns.astype(str))
    else:
        labels = ['bias'] + ['x' + str(i) for i in range(dimension)]

    features = np.column_stack([np.ones((N, 1)), features])
    frame = DataFrame(features, columns=labels)
    return frame
