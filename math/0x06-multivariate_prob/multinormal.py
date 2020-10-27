#!/usr/bin/env python3
""" Multinormal Class """
import numpy as np


class MultiNormal():
    """"""
    def __init__(self, data):
        """ Constructor

            - data is a numpy.ndarray of shape (d, n) containing the
              data set:
            - n is the number of data points
            - d is the number of dimensions in each data point
            - If data is not a 2D numpy.ndarray, raise a TypeError with
              the message data must be a 2D numpy.ndarray
            - If n is less than 2, raise a ValueError with the message
              data must contain multiple data points
        """
        if type(data) is not np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = data.mean(axis=1, keepdims=True)
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (n - 1)
