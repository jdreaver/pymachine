import numpy as np
import cvxopt
from cvxopt.solvers import qp
from cvxopt import matrix

cvxopt.solvers.options['show_progress'] = False

def linsep_svm(X, y):
    """Computes svm for linearly separable data."""
    N = X.shape[0]
    P = matrix(np.dot(X, X.T) * np.outer(y, y))
    q = matrix(-1.0, (N, 1))
    G = matrix(np.diag(np.ones(N) * -1))
    #h = matrix(0.0, (N, 1))
    h = matrix(np.zeros(N))
    A = matrix(y, (1, N))
    b = matrix(0.0)
    solution = qp(P, q, G, h, A, b)
    alpha = np.ravel(solution['x'])

    w = np.sum(alpha*y*X.T, axis=1)
    support_vectors = np.where(alpha > 0.01)[0]

    #b_index = np.argmax(alpha)
    #b = 1.0/y[b_index] - np.dot(w, X[b_index])
    ## Compute b using Bishop formula
    X_sv = X[support_vectors]
    y_sv = y[support_vectors]
    b = np.mean(y_sv - np.dot(X_sv, w))
    return (np.insert(w, 0, b), support_vectors)
    
