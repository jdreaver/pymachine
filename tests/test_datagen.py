"""Tests for pymachine.datagen"""

import numpy as np

from pymachine import datagen

def test_bounds():
    bad_bounds = [-1, 1, 1]
    try:
        datagen.infer_dimension(bad_bounds)
    except AssertionError:
        pass

def test_random_gens():
    for dimension in range(2, 5):
        bounds = datagen.unit_bounds(dimension)
        random_points = datagen.random_plane_points(4, bounds)
        assert np.array_equal(random_points.shape, (4, dimension))
        
        random_line = datagen.random_hyperplane(bounds)
        assert len(random_line) == dimension + 1

def test_linearly_separable():
    "Maybe stick a PLA solver in here too?"
    num_points = 100
    for dimension in range(2, 5):
        bounds = datagen.unit_bounds(dimension)
        (X, y, weights) = datagen.random_linearly_separable_data(num_points, 
                                                                 bounds)
        X_in = np.column_stack([np.ones((num_points, 1)), X])
        assert np.all(y == np.sign(np.dot(X_in, weights)))
        

    
