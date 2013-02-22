"""This modules helps visualize the output of learning algorithms."""

import numpy as np
import matplotlib.pyplot as plt

def decision_boundary_2D(xmin, xmax, xdelta, ymin, ymax, ydelta, decision_fn):

    """Computes decision boundary over 2D grid.
    
    The decision function is used to classify points inside the grid
    specified by the other parameters. The decision function needs to
    be able to take a N by 2 numpy array of coordinates and output the
    labels in {-1, 1}.

    Args:
       xmin: minimum x coordinate
       xmax: maximum x coordinate
       xdelta: change in x coordinate
       ymin: minimum y coordinate
       ymax: maximum y coordinate
       ydelta: change in y coordinate
       decision_fn: function to evaluate set of points in grid.
    
    Returns:

    """

    x = np.arange(xmin, xmax, xdelta)
    y = np.arange(ymin, ymax, ydelta)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.flatten(), Y.flatten()]).T

    labels = decision_fn(grid)
    Z = labels.reshape((len(y), len(x)))
    return (X, Y, Z)

def weights_to_mxb_2D(w, bounds):
    bounds = np.array(bounds)
    x = bounds[:2]
    y = -x*w[1]/w[2] - w[0]/w[2]
    return (x,y)
