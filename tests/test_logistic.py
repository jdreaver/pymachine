"""Unit tests for logistic regression."""

import numpy as np

from pymachine import logistic

def test_computations():
    y = np.array([-1, 1])
    X = np.array([[1, -3, 2],
                  [1, -1, 1]])
    w = np.array([-2, 1, 1.5])
    assert np.allclose(logistic.logistic_gradient(X[0], y[0], w),
                       np.array([0.1192, -0.3576, 0.2384]), atol=0.001)
    assert np.allclose([logistic.logistic_error(X, y, w)],
                       [0.91417064])
