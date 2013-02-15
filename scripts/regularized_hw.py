"""Solution for CalTech ML homework 6."""

import numpy as np
import csv
import matplotlib.pyplot as plt

from pymachine.linreg import weight_decay_regression, linear_regression
from pymachine.error_measure import linear_error
from pymachine.visualization import decision_boundary_2D

def train_test_data(data_file):
    X = []
    y = []
    reader = csv.reader(open(data_file), delimiter=' ', skipinitialspace=True)
    for line in reader:
        X.append([float(i) for i in line[:2]])
        y.append(int(float(line[2])))

    return np.array(X), np.array(y)

def get_data():
    train_file = 'scripts/in.dta'
    test_file = 'scripts/out.dta'

    (X_train, y_train) = train_test_data(train_file)
    (X_test, y_test) = train_test_data(test_file)
    return (X_train, y_train, X_test, y_test)

def transform(X):
    return np.array([[1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], 
                      abs(x[0]-x[1]), abs(x[0]+x[1])]
                     for x in X]) 
    
def plot_nonreg():
    (X_train_raw, y_train, X_test_raw, y_test) = get_data()
    X_train = transform(X_train_raw)
    X_test = transform(X_test_raw)

    w_nonreg = linear_regression(X_train, y_train)

    def plot_decision_fn(X):
        X_trans = transform(X)
        return np.sign(np.dot(X_trans, w_nonreg))

    (cont_x, cont_y, cont_z) = decision_boundary_2D(-1, 1, 0.01, -1, 1, 0.01, 
                                                    plot_decision_fn)

    x_plot = X_test_raw[:,0]
    y_plot = X_test_raw[:,1]
    c = np.where(y_test==1, 'r', 'b')
    plt.scatter(x_plot,y_plot, c=c)
    plt.contour(cont_x, cont_y, cont_z, [0], colors='g')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid()
    plt.show()

def answers():
    k_list = np.array([-3, -2, -1, 0, 1, 2, 3])

    (X_train_raw, y_train, X_test_raw, y_test) = get_data()
    X_train = transform(X_train_raw)
    X_test = transform(X_test_raw)

    w_nonreg = linear_regression(X_train, y_train)

    E_in_nonreg = linear_error(X_train, y_train, w_nonreg)
    E_out_nonreg = linear_error(X_test, y_test, w_nonreg)

    N = X_train.shape[0]
    print "Number of points:", N
    print "Non-regularized stats: "
    print "   E_in: ", E_in_nonreg
    print "   E_out:", E_out_nonreg
    print "   wsum: ", np.power(w_nonreg, 2).sum()

    print "Regularized stats: "
    E_in_reg = np.zeros(len(k_list))
    E_out_reg = np.zeros(len(k_list))
    
    for i, k in enumerate(k_list):
        w_reg = weight_decay_regression(X_train, y_train, 10.0**k/N)
        E_in_reg[i] = linear_error(X_train, y_train, w_reg)
        E_out_reg[i] = linear_error(X_test, y_test, w_reg)

        print "   k =", k, "constant =", 10.0**k/N
        print "      E_in:   ", E_in_reg[i]
        print "      E_out:  ", E_out_reg[i]
        print "      wsum:   ", np.power(w_reg, 2).sum()

if __name__ == '__main__':
    pass
