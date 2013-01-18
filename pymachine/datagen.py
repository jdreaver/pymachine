"""
This module holds functions that are used to help generate training data.
The data can either be totally artificial or simply modified real data.
"""

import numpy as np

def random_plane_points(num_points, bounds=[-1, 1, -1, 1]):
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
    bounds = np.asarray(bounds).flatten()
    assert len(bounds) % 2 == 0, "Bad bounding plane given."
    dimension = bounds.length/2

    # Generate points and rescale to fit bounds
    points = np.random.rand(num_points, dimension)
    shifted_points = points - np.array([0.5, 0.5]) + bounds.mean(axis=1)
    scale = bounds[:,1] - bounds[:,0]
    rescaled_points = np.dot(shifted_points, np.diag(scale))

    return rescaled_points
    
    
