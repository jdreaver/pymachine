"""Tests for pymachine.datagen"""

import pymachine.datagen

def test_points():
    bad_bounds = [-1, 1, 1]
    try:
        pymachine.datagen.infer_dimension(bad_bounds)
    except AssertionError:
        pass

    
