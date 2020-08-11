#!/usr/bin/env python3
""" One hot encoding """
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix
        @Y: numpy.ndarray with shape (m,) containing numeric class labels
        @classes: is the maximum number of classes found in Y
    """
    if type(classes) is not int:
        return None
    if classes <= 0:
        return None
    if Y is None or type(Y) is not np.ndarray:
        return None
    if len(Y) == 0:
        return None
    x = np.zeros((classes, len(Y)))
    for i in range(classes):
        r = np.where(Y == i)
        for j in r:
            x[i][j] = 1
    return x
