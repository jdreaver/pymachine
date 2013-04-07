import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold
import csv

from pymachine.error_measure import linear_error

def train_test_data(data_file):
    X = []
    y = []
    reader = csv.reader(open(data_file), delimiter=' ', skipinitialspace=True)
    for line in reader:
        y.append(int(float(line[0])))
        X.append([float(i) for i in line[1:3]])

    return np.array(X), np.array(y)

def get_data():
    train_file = 'scripts/features.train'
    test_file = 'scripts/features.test'

    (X_train, y_train) = train_test_data(train_file)
    (X_test, y_test) = train_test_data(test_file)
    return (X_train, y_train, X_test, y_test)

def get_digits(first, second, X, y):
    first_digits = X[np.where(y == first)]
    first_labels = np.ones(len(first_digits))
    second_digits = np.concatenate([X[np.where(y == digit)] for digit in second])
    second_labels = -1 * np.ones(len(second_digits))
    X_new = np.concatenate((first_digits, second_digits))
    y_new = np.concatenate((first_labels, second_labels))
    return (X_new, y_new)

def one_vs_rest():
    print("Solving problems 2-4")
    C = 0.01
    d = 2
    
    print("C=", C)
    print("Q=", d)

    digits = list(range(10))
    (X_train, y_train, X_test, y_test) = get_data()
    
    for digit in digits:
        other_digits = [i for i in range(10) if i != digit]
        (X_dig, y_dig) = get_digits(digit, other_digits, X_train, y_train)
        clf = svm.SVC(kernel='poly', gamma=1, degree=d, coef0=1, C=C)
        clf.fit(X_dig, y_dig)
        y_pred = clf.predict(X_dig)
        E_in = np.where(y_dig != y_pred, 1.0, 0.0).mean()
        num_sv = sum(clf.n_support_)
        print(digit, "vs rest")
        print("  E_in: ", E_in)
        print("  num_sv", num_sv)

def one_vs_one():
    print("Solving problems 5 and 6")
    
    digit_1 = 1
    digit_2 = 5
    print("Comparing digits", digit_1, "and", digit_2)
    (X_train, y_train, X_test, y_test) = get_data()
    C_list = [0.0001, 0.001, 0.01, 0.1, 1]
    Q_list = [2, 5]
    (X_dig_train, y_dig_train) = get_digits(digit_1, [digit_2], X_train, y_train)
    (X_dig_test, y_dig_test) = get_digits(digit_1, [digit_2], X_test, y_test)
    
    for C in C_list:
        print("  C =", C)
        for d in Q_list:
            print("    Q =", d)
            clf = svm.SVC(kernel='poly', gamma=1, degree=d, coef0=1, C=C)
            clf.fit(X_dig_train, y_dig_train)
            y_pred_train = clf.predict(X_dig_train)
            y_pred_test = clf.predict(X_dig_test)
            E_in = np.where(y_dig_train != y_pred_train, 1.0, 0.0).mean()
            E_out = np.where(y_dig_test != y_pred_test, 1.0, 0.0).mean()
            num_sv = sum(clf.n_support_)
            print("      E_in:  ", E_in)
            print("      E_out: ", E_out)
            print("      num_sv:", num_sv)

def cross_validation_C(C_list, d, K, X, y):
    E_vals = np.zeros((K, len(C_list)))
    for k, (train, val) in enumerate(KFold(len(y), n_folds=K, shuffle=True)):
        (X_train, y_train, X_val, y_val) = (X[train], y[train], X[val], y[val])
        for i, C in enumerate(C_list):
            clf = svm.SVC(kernel='poly', gamma=1, degree=d, coef0=1, C=C)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            E_val = np.where(y_pred != y_val, 1.0, 0.0).mean()
            E_vals[k, i] = E_val
    E_cv = E_vals.mean(axis=0)
    return E_cv

def one_vs_one_cv():
    print("Solving problems 7 and 8")
    
    digit_1 = 1
    digit_2 = 5
    K = 10
    C_list = [0.0001, 0.001, 0.01, 0.1, 1]
    Q = 2
    num_experiments = 100

    print("Comparing digits", digit_1, "and", digit_2)

    (X_train, y_train, X_test, y_test) = get_data()
    (X_dig_train, y_dig_train) = get_digits(digit_1, [digit_2], X_train, y_train)
    (X_dig_test, y_dig_test) = get_digits(digit_1, [digit_2], X_test, y_test)
    
    selected_counts = np.zeros(len(C_list))
    sum_E_cv = np.zeros(len(C_list))

    for i in range(num_experiments):
        print(i+1, "/", num_experiments)
        E_cv = cross_validation_C(C_list, Q, K, X_dig_train, y_dig_train)
        best_index = np.argmin(E_cv)
        #print "  E_cv:", E_cv
        #print "  min :", best_index
        selected_counts[best_index] += 1
        sum_E_cv += E_cv

    E_cv = sum_E_cv/num_experiments
    best_C_index = np.argmax(selected_counts)
    best_C = C_list[best_C_index]
    best_E_cv = E_cv[best_C_index]
    print("Times C selected:")
    for i, C in enumerate(C_list):
        print("  C =", C)
        print("    E_cv =", E_cv[i])
        print("    num  =", int(selected_counts[i]))
    print("Best C value:", best_C)
    print("Average E_cv:", best_E_cv)
        
        
def rbf_kernel():
    print("Solving problems 9 and 10")
    
    digit_1 = 1
    digit_2 = 5
    C_list = [0.01, 1, 100, 10**4, 10**6]

    print("Comparing digits", digit_1, "and", digit_2)

    (X_train, y_train, X_test, y_test) = get_data()
    (X_dig_train, y_dig_train) = get_digits(digit_1, [digit_2], X_train, y_train)
    (X_dig_test, y_dig_test) = get_digits(digit_1, [digit_2], X_test, y_test)

    for i, C in enumerate(C_list):
        clf = svm.SVC(kernel='rbf', gamma=1, C=C)
        clf.fit(X_dig_train, y_dig_train)
        y_pred_train = clf.predict(X_dig_train)
        y_pred_test = clf.predict(X_dig_test)
        E_in = np.where(y_dig_train != y_pred_train, 1.0, 0.0).mean()
        E_out = np.where(y_dig_test != y_pred_test, 1.0, 0.0).mean()
        print("C:", C)
        print("  E_in: ", E_in)
        print("  E_out:", E_out)
