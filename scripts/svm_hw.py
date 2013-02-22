import matplotlib.pyplot as plt
import numpy as np

from pymachine.svm import linsep_svm
from pymachine.datagen import random_linearly_separable_data
from pymachine.error_measure import linear_randomized_Eout
from pymachine.perceptron import PLA
from pymachine.visualization import weights_to_mxb_2D

bounds = [-1, 1, -1, 1]

def plot_svm():
    num_points = 20
    (X, y, f) = random_linearly_separable_data(num_points, bounds)
    (w_svm, vectors) = linsep_svm(X, y)
    
    x_plot = X[:,0]
    y_plot = X[:,1]
    s = 20*np.ones(num_points)
    s[vectors] = 60
    #markers = np.chararray(num_points)
    #markers[:] = 'o'
    #markers[vectors] = 'D'

    (x_f, y_f) = weights_to_mxb_2D(f, bounds)
    (x_w, y_w) = weights_to_mxb_2D(w_svm, bounds)
    c = np.where(y==1, 'r', 'b')
    plt.scatter(x_plot,y_plot, s, c=c)
    plt.plot(x_f, y_f, 'k-.')
    plt.plot(x_w, y_w, 'b')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid()
    plt.show()

def answers():
    num_points = 10
    num_experiments = 1000
    times_svm_better = 0.0
    total_num_vectors = 0.0

    for i in range(num_experiments):
        (X, y, f) = random_linearly_separable_data(num_points, bounds)
        (w_svm, vectors) = linsep_svm(X, y)
        pla = PLA(bounds=bounds)
        pla.fit(X, y)
        w_pla = pla.weights
        E_svm = linear_randomized_Eout(f, w_svm, bounds)
        E_pla = linear_randomized_Eout(f, w_pla, bounds)
        if E_svm < E_pla:
            times_svm_better += 1
        total_num_vectors += len(vectors)

    print "Number of points used:            ", num_points
    print "Proportion of times SVM beats PLA:", times_svm_better/num_experiments
    print "Average number of support vectors:", total_num_vectors/num_experiments

if __name__ == '__main__':
    plot_svm()
    #answers()









