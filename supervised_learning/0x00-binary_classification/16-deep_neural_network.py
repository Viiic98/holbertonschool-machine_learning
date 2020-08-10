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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if type(layers[l]) is not int or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w = "W" + str(l + 1)
            b = "b" + str(l + 1)

            if l == 0:
                self.weights[w] = np.random.randn(layers[l], nx)\
                                  * np.sqrt(2. / nx)
            else:
                self.weights[w] = np.random.randn(layers[l], layers[l - 1])\
                                  * np.sqrt(2 / layers[l - 1])
            self.weights[b] = np.zeros((layers[l], 1))
