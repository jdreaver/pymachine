import numpy as np

from pymachine.linreg import linear_regression
from pymachine.error_measure import linear_error
from regularized_hw import get_data, transform


def answers():
    ks = [3, 4, 5, 6, 7]

    (X_train_raw, y_train, X_test_raw, y_test) = get_data()
    X_train = transform(X_train_raw)
    X_test = transform(X_test_raw)
    X_val = X_train[25:]
    y_val = y_train[25:]
    X_train = X_train[:25]
    y_train = y_train[:25]

    print "Training:", X_train.shape[0], "Validation:", X_val.shape[0]
    for k in ks:
        K = k+1
        (E_val, E_out) = train_and_eval(X_train[:,:K], y_train, X_val[:,:K], y_val,
                                        X_test[:,:K], y_test)
        print "  k =", k
        print "    E_val:", E_val
        print "    E_out:", E_out

    (X_train, y_train, X_val, y_val) = (X_val, y_val, X_train, y_train)
    print "Training:", X_train.shape[0], "Validation:", X_val.shape[0] # Switch the two
    for k in ks:
        K = k+1
        (E_val, E_out) = train_and_eval(X_train[:,:K], y_train, X_val[:,:K], y_val,
                                        X_test[:,:K], y_test)
        print "  k =", k
        print "    E_val:", E_val
        print "    E_out:", E_out

def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test):
    w = linear_regression(X_train, y_train)
    E_val = linear_error(X_val, y_val, w)
    E_out = linear_error(X_test, y_test, w)
    return E_val, E_out

if __name__ == '__main__':
    answers()
