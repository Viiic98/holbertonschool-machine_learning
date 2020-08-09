#!/usr/bin/env python3
""" Neural Network """
import numpy as np


class NeuralNetwork():
    """ Neural network with one hidden layer """
    def __init__(self, nx, nodes):
        """ Class constructor
            @nx: Number of input features
            @nodes: Nodes found in the hidden layer

            Public instance attributes:
            @W1: weights vector for the hidden layer
            @b1: bias for the hidden layer.
            @A1: activated output for the hidden layer
            @W2: weights vector for the output neuron
            @b2: bias for the output neuron
            @A2: activated output for the output neuron (prediction)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
