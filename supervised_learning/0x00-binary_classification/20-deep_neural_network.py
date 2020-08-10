#!/usr/bin/env python3
""" Deep neural network """
import numpy as np


class DeepNeuralNetwork():
    """ defines a deep neural network """
    def __init__(self, nx, layers):
        """ Class constructor

            @nx: number of input features
            @layers: number of nodes in each layer of the network
            @L: number of layers in the neural network
            @cache: dictionary to hold all intermediary values of the network
            @weights: dictionary to hold all weights and biased of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w = "W" + str(i + 1)
            b = "b" + str(i + 1)

            if i == 0:
                self.weights[w] = np.random.randn(layers[i], nx)\
                                  * np.sqrt(2. / nx)
            else:
                self.weights[w] = np.random.randn(layers[i], layers[i - 1])\
                                  * np.sqrt(2 / layers[i - 1])
            self.weights[b] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter function """
        return self.__L

    @property
    def cache(self):
        """ Getter function """
        return self.__cache

    @property
    def weights(self):
        """ Getter function """
        return self.__weights

    def forward_prop(self, X):
        """ Activation function
            Sigmoid Forward propagation
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            a = 'A' + str(i + 1)
            Z = np.dot(self.__weights[w], self.__cache['A' + str(i)])\
                + self.__weights[b]
            self.__cache[a] = 1 / (1 + np.exp(-Z))
        return self.__cache[a], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        error = (-Y * np.log(A)) - ((1-Y) * np.log(1.0000001-A))
        cost = (1 / m) * np.sum(error)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron """
        self.forward_prop(X)
        l = 'A' + str(self.__L)
        A = self.__cache[l]
        return np.round(A).astype(int), self.cost(Y, A)
