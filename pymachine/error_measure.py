"""This module implements error measuring functions."""

import numpy as np

from pymachine.datagen import random_plane_points
from pymachine.logistic import cross_entropy_error

def linear_randomized_Eout(f, g, bounds, N=10000):

    """Computes out of sample error for two linear weights.

    This function compute the out of sample error E_out by generating
    N random points in the given bounds, and finding the percentage difference
    in labels from the two lines defined by f and g.

    Args:
        f, g: two lines defined by weights (with bias term)
        bounds: bounds used to generate random points
        N: number of points to generate

    Returns:
        Proportion of generated points whose labels differ from g to f.
    """

    X = np.hstack([np.ones((N, 1)), random_plane_points(N, bounds)])
    labels_f = np.sign(np.dot(X, f))
    labels_g = np.sign(np.dot(X, g))
    return np.where(labels_f==labels_g, 0, 1.0).mean()

def linear_error(X, y, w):

    """Computes proprotion of X.T*w != y"""

    return np.where(y != np.dot(X, w), 1.0, 0.0).mean()

def cross_entropy_randomized_Eout(f, g, bounds, N=10000):

    """Computes cross entropy out of sample error for linear model.

    The cross entropy error is similar to the linear_Eout error, but
    instead of simply taking the proportion of wrong labels, we use cross
    entropy to get an error.

    Args:
        f, g: two lines defined by weights (with bias term)
        bounds: bounds used to generate random points
        N: number of points to generate

    Returns:
        Cross entropy error from g to f.
    """

    X = np.hstack([np.ones((N, 1)), random_plane_points(N, bounds)])
    labels_f = np.sign(np.dot(X, f))
    return cross_entropy_error(X, labels_f, g)
