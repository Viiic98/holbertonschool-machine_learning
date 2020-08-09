#!/usr/bin/env python3
""" Class Neuron """
import numpy as np


class Neuron():
    """ Class Neuron performing binary classification """
    def __init__(self, nx):
        """ Class constructor """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter attribute """
        return self.__W

    @property
    def b(self):
        """ b getter attribute """
        return self.__b

    @property
    def A(self):
        """ A getter attribute """
        return self.__A

    def forward_prop(self, X):
        """ Forward propagation Sigmoid """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        error = (-Y * np.log(A)) - ((1-Y) * np.log(1.0000001-A))
        cost = (1 / m) * np.sum(error)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron """
        A = self.forward_prop(X)
        return np.round(A).astype(int), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Gradient descent """
        dz = A - Y
        dw = (1 / len(X[0])) * np.matmul(X, np.transpose(dz))
        db = (1 / len(X[0])) * np.sum(dz)
        self.__W = self.__W - (alpha * np.transpose(dw))
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
