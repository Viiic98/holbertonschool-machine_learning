#!/usr/bin/env python3
""" Dropout regularization """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ conducts forward propagation using Dropout

        @X: is a numpy.ndarray of shape (nx, m) containing
            the input data for the network
        @nx: is the number of input features
        @m: is the number of data points
        @weights: is a dictionary of the weights and biases
                  of the neural network
        @L: the number of layers in the network
        @keep_prob: is the probability that a node will be kept
        - All layers except the last should use the tanh activation
          function
        The last layer should use the softmax activation function
        Returns: a dictionary containing the outputs of each layer
                 and the dropout mask used on each layer
    """
    drop_out = {}
    A = X
    drop_out['A0'] = A / keep_prob
    for i in range(L):
        if i > 0:
            d = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob)\
                .astype(int)
            drop_out['D' + str(i)] = d
            A = np.multiply(A, d)
        A /= keep_prob
        Z = np.dot(weights['W' + str(i + 1)], A) + weights['b' + str(i + 1)]
        # Activation
        # Softmax
        if i + 1 == L:
            exp = np.exp(Z)
            A = exp / np.sum(exp, axis=0, keepdims=True)
        else:
            # Tanh Activation
            A = np.tanh(Z)
        drop_out['A' + str(i + 1)] = A
    return drop_out
