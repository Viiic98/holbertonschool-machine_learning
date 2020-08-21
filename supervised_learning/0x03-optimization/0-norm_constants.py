#!/usr/bin/env python3
""" Standard normalization """
import numpy as np


def normalization_constants(X):
    """ calculates the normalization (standardization)
        constants of a matrix

        @X: is the numpy.ndarray of shape (m, nx) to normalize
            @m: is the number of data points
            @nx: is the number of features
        Formulas:
            Mean = (1 / m) * Sigma(X)
            Variance = (1 / m) * Sigma((X - Mean) ** 2)
        Returns: the mean and standard deviation of each feature
    """
    m = len(X)
    mean = (1 / m) * sum(X)
    variance = (1 / m) * sum((X - mean) ** 2)
    stdv = np.sqrt(variance)
    return mean, stdv
