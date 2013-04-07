"""This module implements radial basis functions."""

import numpy as np
from scipy.spatial.distance import cdist
import sklearn.cluster

from pymachine.datagen import random_plane_points
from pymachine.linreg import linear_regression

def cluster_centers(X, k, bounds):
    """Returns mu for k-means on X."""

    mu = random_plane_points(k, bounds)
    kmeans = sklearn.cluster.KMeans(init=mu)
    kmeans.fit(X)

    # Restart if a mean has no associated points
    if len(set(kmeans.labels_)) < k:
        return cluster_centers(X, k, bounds)

    return kmeans.cluster_centers_


def rbf_matrix(gamma, X, mu=None):

    if mu is None:
        mu = X

    outer_distance = cdist(X, mu, 'sqeuclidean')
    return np.exp(-gamma * outer_distance)


class rbfClassifier(object):
    """
    """
    
    def __init__(self, k, gamma, bounds, bias=True):
        self.k = k
        self.gamma = gamma
        self.bounds = bounds
        self.weights = None
        self.bias = bias
        
    def fit(self, X, y, mus=None):
        self.mus = mus
        if mus is None:
            self.mus = cluster_centers(X, self.k, self.bounds)    
        matrix = rbf_matrix(self.gamma, X, self.mus)
        if self.bias is True:
            matrix = np.hstack([np.ones((X.shape[0], 1)), matrix])
        
        self.weights = linear_regression(matrix, y)

    def predict(self, X):
        if self.weights == None:
            print("NEED TO FIT")
        
        matrix = rbf_matrix(self.gamma, X, self.mus)
        if self.bias is True:
            matrix = np.hstack([np.ones((X.shape[0], 1)), matrix])
        return np.sign(np.dot(matrix, self.weights))
