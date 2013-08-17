import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

from pymachine.datagen import random_plane_points
from pymachine.error_measure import linear_error
from pymachine.linreg import weight_decay_regression
from pymachine.rbf import rbfClassifier
from pymachine.visualization import weights_to_mxb_2D
from .soft_svm_hw import get_data as get_digit_data
from .soft_svm_hw import get_digits

## Regularized Least Squares on Digits (7, 8, 9, 10)

def transform(X):
    return np.array([[1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2]
                     for x in X]) 

    
def one_vs_rest():
    print("Solving problems 7-9")
    decay = 1
    print("decay=", decay)

    digits = list(range(10))
    (X_train, y_train, X_test, y_test) = get_digit_data()
    #(X_train_trans, X_test_trans) = (transform(X_train), transform(X_test))

    for digit in digits:
        other_digits = [i for i in range(10) if i != digit]
        (X_dig_train, y_dig_train) = get_digits(digit, other_digits, X_train, y_train)
        (X_dig_test, y_dig_test) = get_digits(digit, other_digits, X_test, y_test)
        w_lin = weight_decay_regression(X_dig_train, y_dig_train, decay)
        E_in_lin = linear_error(X_dig_train, y_dig_train, w_lin)
        E_out_lin = linear_error(X_dig_test, y_dig_test, w_lin)

        (X_train_nonlin, X_test_nonlin) = (transform(X_dig_train), transform(X_dig_test))
        w_nonlin = weight_decay_regression(X_train_nonlin, y_dig_train, decay)
        E_in_nonlin = linear_error(X_train_nonlin, y_dig_train, w_nonlin)
        E_out_nonlin = linear_error(X_test_nonlin, y_dig_test, w_nonlin)

        print(digit, "vs rest")

        print("  E_in_lin:     ", E_in_lin)
        print("  E_out_lin:    ", E_out_lin)
        print("  E_in_nonlin:  ", E_in_nonlin)
        print("  E_out_nonlin: ", E_out_nonlin)

def one_vs_one():
    print("Solving problem 10")
    digit_1 = 1
    digit_2 = 5
    print("Comparing digits", digit_1, "and", digit_2)
    (X_train, y_train, X_test, y_test) = get_digit_data()
    lamba_list = [0.01, 1]
    (X_dig_train, y_dig_train) = get_digits(digit_1, [digit_2], X_train, y_train)
    (X_dig_test, y_dig_test) = get_digits(digit_1, [digit_2], X_test, y_test)
    (X_dig_train, X_dig_test) = (transform(X_dig_train), transform(X_dig_test))

    for decay in lamba_list:
        print("lambda =", decay)
        w = weight_decay_regression(X_dig_train, y_dig_train, decay)
        E_in = linear_error(X_dig_train, y_dig_train, w)
        E_out = linear_error(X_dig_test, y_dig_test, w)
        print("  E_in: ", E_in)
        print("  E_out:", E_out)

 
## SVM for problems 12 and 13

def simple_svm():
    print("Solving problems 12 and 13")

    X = np.array([[1,  0], [0, 1], [0, -1],
                  [-1, 0], [0, 2], [0, -2],
                  [-2, 0]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])
    
    def svm_transform(X):
        return np.array([[x[1]**2-2*x[0]-1, x[0]**2-2*x[1]+1] for x in X])

    X_t = svm_transform(X)

    clf = svm.SVC(kernel='poly', gamma=1, degree=2, coef0=1, C=10**10)
    clf.fit(X, y)
    print("num vectors: ", sum(clf.n_support_))
    w_test = np.array([-0.5, 1, 0])

    # Plot
    plt.scatter(X_t[:, 0], X_t[:, 1], c=np.where(y==1, 'r', 'b'))
    ax = plt.gca()
    bounds = np.concatenate([ax.get_xlim(), ax.get_ylim()])
    (x_t, y_t) = weights_to_mxb_2D(w_test, bounds)
    plt.grid()
    plt.plot(x_t, y_t, 'r--')
    plt.show()


## RBF for 14 - 19

# def rbf_answers():
#     num_experiments = 10
#     num_points = 100
#     k_list = [9, 12]
#     gamma_list = [1.5, 2]
#     k_gamma = itertools.product(k_list, gamma_list)

#     num_kernel_beat_reg = np.zeros(len(k_list) * len(gamma_list))
#     not_linsep = np.zeros(len(gamma_list))  # Only for kernel
#     for experiment in num_experiments:
#         for i, gamma in enumerate(gamma_list):
#             for j, k in enumerate(k_list):
#                 pass

def target_fn(X):
    return np.sign([x[1] - x[0] + 0.25 * np.sin(np.pi * x[0]) for x in X])

def rbf_answers(k=9, gamma=1.5):
    num_experiments = 500
    num_points = 100
    bounds = [-1, 1, -1, 1]
    
    num_kernel_beat_reg = 0.0
    not_linsep = 0.0
    num_rbf_linsep = 0.0
    sum_E_in_svm = 0.0
    sum_E_out_svm = 0.0
    sum_E_in_rbf = 0.0
    sum_E_out_rbf = 0.0
    for experiment in range(num_experiments):
        #print experiment + 1, "/", num_experiments
        X = random_plane_points(num_points, bounds)
        y = target_fn(X)
        svm_clf = svm.SVC(kernel='rbf', gamma=gamma, C=10**10)
        svm_clf.fit(X, y)
        rbf_clf = rbfClassifier(k, gamma, bounds, bias=True)
        rbf_clf.fit(X, y)
        E_in_svm = np.where(svm_clf.predict(X) != y, 1.0, 0.0).mean()
        E_in_rbf = np.where(rbf_clf.predict(X) != y, 1.0, 0.0).mean()
        X_test = random_plane_points(10000, bounds)
        y_test = target_fn(X_test)
        E_out_svm = np.where(svm_clf.predict(X_test) != y_test, 1.0, 0.0).mean()
        E_out_rbf = np.where(rbf_clf.predict(X_test) != y_test, 1.0, 0.0).mean()

        # Statistics
        if E_in_svm != 0:    # Throw out run
            not_linsep += 1
            continue
        if E_in_rbf == 0:
            num_rbf_linsep += 1
        if E_out_svm < E_out_rbf:
            num_kernel_beat_reg += 1
        sum_E_in_svm += E_in_svm
        sum_E_out_svm += E_out_svm
        sum_E_in_rbf += E_in_rbf
        sum_E_out_rbf += E_out_rbf

    good_runs = num_experiments - not_linsep
    print("k =", k, "gamma =", gamma)
    print("Not linearly separable:     ", not_linsep/num_experiments)
    print("Kernel beats regular:       ", num_kernel_beat_reg/good_runs)
    print("Regular linearly separable: ", num_rbf_linsep/good_runs)
    print("Average value of E_in_svm:  ", sum_E_in_svm/good_runs)
    print("Average value of E_out_svm: ", sum_E_out_svm/good_runs)
    print("Average value of E_in_rbf:  ", sum_E_in_rbf/good_runs)
    print("Average value of E_out_rbf: ", sum_E_out_rbf/good_runs)
