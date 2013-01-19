"""
This module holds functions that are used to help generate training data.
The data can either be totally artificial or simply modified real data.
"""

import numpy as np


def infer_dimension(bounds):

    """Computes the dimension of bounds.

    Computes the dimension of the bounds and reshapes the bounds into a 
    dimension by two array.

    Args:
        bounds: plane defined by [x1_min, x1_max, x2_min, x2_max, ...]

    Returns:
        Reshaped bounds and dimension of bounds.
    """

    bounds = np.asarray(bounds).flatten()
    assert len(bounds) % 2 == 0, "Bad bounding plane given."
    dimension = len(bounds)/2
    return (bounds.reshape((dimension, -1)), dimension)


def unit_bounds(dimension):

    """Generates unit bounds for given dimension."""

    return [-1.0, 1.0] * dimension


def random_plane_points(num_points, bounds):

    """Generates random points on plane.

    Generates points uniformly on a plane defined by the bounds
    variable. The dimension of the plane is calculated by 

    Args:
        num_points: number of points to generate
        bounds: plane defined by [x1_min, x1_max, x2_min, x2_max, ...]

    Returns: 
        A matrix of shape num_points by dimension holding the
        randomly generated points.

    """

    # Infer dimension of data from bounds
    (bounds, dimension) = infer_dimension(bounds)

    # Generate points and rescale to fit bounds
    points = np.random.rand(num_points, dimension)
    unit_mean = [0.5] * dimension
    shifted_points = points - unit_mean + bounds.mean(axis=1)
    scale = bounds[:, 1] - bounds[:, 0]
    rescaled_points = np.dot(shifted_points, np.diag(scale))

    return rescaled_points


def random_hyperplane(bounds):

    """Creates random hyperplane through points in bounds.

    Randomly generates points within bounds and returns the set of
    weights w such that dot(w,x)= 0 for all points x that lie on the
    resulting hyperplane. Note that x needs a constant 1 term to
    account for bias term in weights. For example, the point (2.5, 1.1) is
    x = [1, 2.5, 1.1]

    Args:
        bounds: plane defined by [x1_min, x1_max, x2_min, x2_max, ...]

    Returns:
        The hyperplane within bounds.

    """

    (bounds, dimension) = infer_dimension(bounds)
    intercept_points = random_plane_points(dimension, bounds)

    # Solve for weights. Bias is arbitrary
    bias = 1
    line_weights = np.linalg.solve(intercept_points, 
                                   np.ones((dimension, 1)) * - bias)
    line_weights *= np.sign(np.random.rand(1) - 0.5) # Randomize direction

    return np.append(bias, line_weights)
    
    
