#!/usr/bin/env python3
""" Gradient descent with L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network using
        gradient descent with L2 regularization

        @Y: is a one-hot numpy.ndarray of shape (classes, m) that
            contains the correct labels for the data
            @classes: is the number of classes
            @m: is the number of data points
        @weights: is a dictionary of the weights and biases of the
                  neural network
        @cache: is a dictionary of the outputs of each layer of the
                neural network
        @alpha: is the learning rate
        @lambtha: is the L2 regularization parameter
        @L: is the number of layers of the network
        The neural network uses tanh activations on each layer except
        the last, which uses a softmax activation
        The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    # cost
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        a = 'A' + str(i - 1)
        w = 'W' + str(i)
        b = 'b' + str(i)
        A = cache[a]
        dw = (1 / m) * np.matmul(dz, np.transpose(A)) + ((lambtha / m) *
                                                         weights[w])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(np.transpose(weights[w]), dz) * A * (1 - A)
        weights[w] = weights[w] - (alpha * dw)
        weights[b] = weights[b] - (alpha * db)
