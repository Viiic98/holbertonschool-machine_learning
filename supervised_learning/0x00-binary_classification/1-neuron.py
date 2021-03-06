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
