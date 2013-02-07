"""This module implements error measuring functions."""

import numpy as np

from datagen import random_plane_points

def linear_Eout(f, g, bounds, N=10000):

    """Computes out of sample error for two linear weights.

    This function compute the out of sample error E_out by generating
    N random points in the given bounds, and finding the percentage difference
    in labels from the two lines defined by f and g.

    Args:
        f, g: two lines defned by weights (with bias term)
        bounds: bounds used to generate random points
        N: number of points to generate

    Returns:
        Proportion of generated points whose labels differ from g to f.
    """

    X = np.hstack([np.ones((N, 1)), random_plane_points(N, bounds)])
    labels_f = np.sign(np.dot(X, f))
    labels_g = np.sign(np.dot(X, g))
    return np.where(labels_f==labels_g, 0, 1.0).mean()
