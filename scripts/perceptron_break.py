"""We want to find the breakpoint of the perceptron in 3 dimensions."""

import numpy as np
from itertools import product

from pymachine.perceptron import PLA
from pymachine.datagen import random_plane_points

def break_point_pla(dimension):
    num_experiments = 3
    max_iter = 1000
    num_points = 2
    pla = PLA(dimension=dimension)
    while True:
        dichotomies = np.array(list(product([-1, 1], repeat=num_points)))
        for i in range(num_experiments):
            passed = test_all_dichotomies(num_points, max_iter, dichotomies, pla)
            if passed: break
        if passed:
            num_points += 1
        else:
            break
    return num_points

def test_all_dichotomies(num_points, max_iter, dichotomies, pla):
    X = random_plane_points(num_points, pla.bounds)
    passed = True
    for i, dichotomy in enumerate(dichotomies):
        pla.fit(X, dichotomy, maxiter=max_iter)
        if pla.num_iters == max_iter:
            #print dichotomy, np.sign(np.dot(pla.X, pla.weights)), pla.num_iters
            passed = False
            break
    return passed






















