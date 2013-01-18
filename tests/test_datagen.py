"""Tests for pymachine.datagen"""

import pymachine.datagen

def test_points():
    bad_bounds = [-1, 1, 1]
    try:
        pymachine.datagen.random_plane_points(3, bad_bounds)
    except AssertionError:
        pass

    
