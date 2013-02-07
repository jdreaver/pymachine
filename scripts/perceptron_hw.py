"""Answers to homework problems pertaining to perceptrons."""

import numpy as np

from pymachine.perceptron import PLA

bounds = [-1, 1, -1, 1]

def calculate_ans(num_points=10, num_experiments=1000, damping=None):
    num_iterations = np.empty(num_experiments)
    overlap_error = np.empty(num_experiments)
    #all_g = np.empty((num_experiments, 3))
    #all_f = np.empty((num_experiments, 3))
    for i in range(num_experiments):
        (X, y, f) = create_linear_random_data(num_points)
        weights = np.zeros(3)
        old_weights = np.ones(3)
        n = 0
        while not np.array_equal(weights, old_weights):
            old_weights = weights
            weights = update_weights(weights, X, y, damping)
            n += 1
        num_iterations[i] = n
        overlap_error[i] = compute_overlap_error(f, weights)
        #print weights_to_mxb(f), weights_to_mxb(weights)
        #all_g[i] = weights
        #all_f[i] = f

    print "Average iterations to converge:", num_iterations.mean()
    print "Average probability of error:", overlap_error.mean()
