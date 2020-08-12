#!/usr/bin/env python3
""" Deep neural network """
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        i = 'A' + str(self.__L)
        A = self.__cache[i]
        return np.round(A).astype(int), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Gradient descent
            @Y: Labels for the input data
            @cache: dictionary containing all the intermediary
                    values of the network
            @alpha: Learning rate
        """
        m = len(Y[0])
        dz = self.__cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            a = 'A' + str(i - 1)
            w = 'W' + str(i)
            b = 'b' + str(i)
            A = self.__cache[a]
            dw = (1 / m) * np.dot(dz, np.transpose(A))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(np.transpose(self.__weights[w]), dz) * A * (1 - A)
            self.__weights[w] = self.__weights[w] - (alpha * dw)
            self.__weights[b] = self.__weights[b] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the deep neural network

            @X: Input data
            @Y: Correct labels for the input data
            @iterations: Number of iterations to train over
            @alpha: Learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if type(step) is not int:
            raise TypeError("step must be an integer")
        if step > iterations:
            raise ValueError("step must be positive and <= iterations")
        it_x = []
        cost_y = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            if (i % step) == 0:
                a = 'A' + str(self.__L)
                cost = self.cost(Y, self.__cache[a])
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    it_x.append(i)
                    cost_y.append(cost)
            self.gradient_descent(Y, self.__cache, alpha)

        plt.plot(it_x, cost_y)
        plt.title("Training Cost")
        plt.xlabel("iterations")
        plt.ylabel("cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if len(filename.split('.')) == 1:
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        with open(filename, 'rb') as f:
            a = pickle.load(f)
            return a
