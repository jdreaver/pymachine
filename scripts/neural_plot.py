import matplotlib.pyplot as plt
import numpy as np

from pymachine.neural import (NeuralNetwork, SGD, compute_activations, modify_weights,
                              network_gradient)
from pymachine.visualization import decision_boundary_2D
from regularized_hw import get_data


def wtf():
    # layer_sizes = [2, 6, 1]
    # print [(layer_sizes[i], layer_sizes[i + i]) for i in range(2)]
    x = [2, 6, 1, 5]
    print x, range(len(x) - 1)
    #print [(x[i], x[i + i]) for i in range(len(x) - 1)]
    #print [(i + 1) for i in range(len(x) - 1)]
    print [(x[i], x[i + 1]) for i in range(len(x) - 1)]

def x_squared_plot():
    layer_sizes = [2, 6, 6, 1]
    nn = NeuralNetwork(layer_sizes)
    X = np.arange(-1, 1, 0.05)
    y = np.power(X, 2)
    nn = SGD(X, y, nn, max_iter = 1500, printing=True)

    out = [compute_activations(x, nn)[-1] for x in X]

    plt.scatter(X, y)
    plt.plot(X, out, c='r')
    plt.grid()
    plt.show()

def caltech_classification_data():
    """NEED TO FIGURE OUT NN FOR CLASSIFICATION"""
    (X_train_raw, y_train, X_test_raw, y_test) = get_data()

    nn = NeuralNetwork([3, 4, 1])

    def plot_decision_fn(X):
        out = [compute_activations(x, nn)[-1] for x in X]
        return out

    (cont_x, cont_y, cont_z) = decision_boundary_2D(-1, 1, 0.0025, -1, 1, 0.0025, 
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


def test_gradient():
    nn = NeuralNetwork([2, 3, 1])

    X = np.arange(-1, 1, 0.01)
    y = np.power(X, 2)

    error1 = np.abs(compute_activations(X[0], nn)[-1] - y[0])
    derivs = network_gradient(X[0], y[0], nn)
    nn.weights = modify_weights(nn, derivs, 0.1)
    error2 = np.abs(compute_activations(X[0], nn)[-1] - y[0])

    print error1, error2
    

if __name__ == '__main__':
    #test_gradient()
    x_squared_plot()
    #wtf()










