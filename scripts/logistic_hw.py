from __future__ import division
import numpy as np

from pymachine.datagen import random_linearly_separable_data
from pymachine.logistic import logistic_gradient_descent
from pymachine.error_measure import cross_entropy_Eout

import matplotlib.pyplot as plt

bounds = [-1, 1, -1, 1]

def linsep_logistic(num_points=100, num_experiments=100, tol=0.01, eta=0.01, max_iter=1000):
    
    all_epochs = np.zeros(num_experiments)
    all_E_out = np.zeros(num_experiments)

    for i in range(num_experiments):
        (X, y, f) = random_linearly_separable_data(num_points, bounds)
        (weights, num_epochs) = logistic_gradient_descent(X, y, tol, eta, max_iter)
        E_out = cross_entropy_Eout(f, weights, bounds)
        all_epochs[i] = num_epochs
        all_E_out[i] = E_out
        print i+1, num_epochs, E_out

    return all_epochs.mean(), all_E_out.mean()


def test(num_points=100, tol=0.01, eta=0.01, max_iter=2000):
    (X, y, f) = random_linearly_separable_data(num_points, bounds)
    (w, num) = logistic_gradient_descent(X, y, tol, eta, max_iter)

    print num, cross_entropy_Eout(f, w, bounds)
    
    positives = X[np.where(y==1)]
    negatives = X[np.where(y==-1)]

    x_p = positives[:,0]
    y_p = positives[:,1]
    x_n = negatives[:,0]
    y_n = negatives[:,1]
    (w_x, w_y) = (bounds[:2], -np.array(bounds[:2])*w[1]/w[2] - w[0]/w[2])
    (f_x, f_y) = (bounds[:2], -np.array(bounds[:2])*f[1]/f[2] - f[0]/f[2])
    plt.plot(w_x, w_y, c='b')
    plt.plot(f_x, f_y, c='k')
    plt.scatter(x_p, y_p, c='b', marker='o')
    plt.scatter(x_n, y_n, c='r', marker='o')
    plt.xlim(bounds[:2])
    plt.ylim(bounds[2:4])
    plt.show()
    
    
    
