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
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Getter function """
        return self.__W1

    @property
    def b1(self):
        """ Getter function """
        return self.__b1

    @property
    def A1(self):
        """ Getter function """
        return self.__A1

    @property
    def W2(self):
        """ Getter function """
        return self.__W2

    @property
    def b2(self):
        """ Getter function """
        return self.__b2

    @property
    def A2(self):
        """ Getter function """
        return self.__A2

    def forward_prop(self, X):
        """ Activation function
            Sigmoid Forward propagation
        """
        Z = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z))
        Z = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        error = (-Y * np.log(A)) - ((1-Y) * np.log(1.0000001-A))
        cost = (1 / m) * np.sum(error)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron """
        A1, A2 = self.forward_prop(X)
        return np.round(A2).astype(int), self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Gradient descent
            @X: Input data
            @Y: Labels for the input data
            @A1: Output of the hidden layer
            @A2: Predicted output
        """
        dz2 = A2 - Y
        dw2 = (1 / len(X[0])) * np.dot(dz2, np.transpose(A1))
        db2 = (1 / len(X[0])) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(np.transpose(self.__W2), dz2) * A1 * (1 - A1)
        dw1 = (1 / len(X[0])) * np.dot(dz1, np.transpose(X))
        db1 = (1 / len(X[0])) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
