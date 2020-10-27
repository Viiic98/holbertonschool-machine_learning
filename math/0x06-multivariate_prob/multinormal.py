#!/usr/bin/env python3
""" Multinormal Class """
import numpy as np


class MultiNormal():
    """ MultiNormal Class """
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

    def pdf(self, x):
        """calculates the PDF at a data point

            - x is a numpy.ndarray of shape (d, 1) containing the data point
              whose PDF should be calculated
                - d is the number of dimensions of the Multinomial instance
            - If x is not a numpy.ndarray, raise a TypeError with the message
              x must be a numpy.ndarray
            - If x is not of shape (d, 1), raise a ValueError with the message
              x must have the shape ({d}, 1)
            Returns the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        if x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        inv = np.linalg.inv(self.cov)
        det = np.linalg.det(self.cov)
        a = 1 / np.sqrt((((2 * np.pi)**(x.shape[0]) * det)))
        inv = np.matmul((x - self.mean).T, inv)
        b = np.exp(-(1/2) * ((np.matmul(inv, (x - self.mean)))))
        pdf = a * b
        return pdf[0][0]
