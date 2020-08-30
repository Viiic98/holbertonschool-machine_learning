#!/usr/bin/env python3
""" Dropout with gradient descent """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights of a neural network with Dropout regularization
        using gradient descent

        @Y: is a one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
        @classes: is the number of classes
        @m: is the number of data points
        @weights: is a dictionary of the weights and biases of the neural
                  network
        @cache: is a dictionary of the outputs and dropout masks of each
                layer of the neural network
        @alpha: is the learning rate
        @keep_prob: is the probability that a node will be kept
        @L: is the number of layers of the network
        All layers use thetanh activation function except the last, which
        uses the softmax activation function
        The weights of the network should be updated in place
    """
    m = Y.shape[1]
    # cost
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        a = 'A' + str(i - 1)
        w = 'W' + str(i)
        b = 'b' + str(i)
        A = cache[a]
        dw = (1 / m) * np.dot(dz, np.transpose(A))
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if 'D' + str(i - 1) in cache:
            mask = cache['D' + str(i - 1)]
            dz = np.matmul(np.transpose(weights[w]), dz) \
                * (1 - (A**2)) * (mask / keep_prob)
        else:
            dz = np.matmul(np.transpose(weights[w]), dz) * A * (1 - A)
        weights[w] = weights[w] - (alpha * dw)
        weights[b] = weights[b] - (alpha * db)
