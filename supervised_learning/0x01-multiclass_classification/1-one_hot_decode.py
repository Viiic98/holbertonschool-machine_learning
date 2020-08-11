#!/usr/bin/env python3
""" One hot decoder """
import numpy as np


def one_hot_decode(one_hot):
    """ converts a one-hot matrix into a vector of labels
        @one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
            @classes is the maximum number of classes
            @m is the number of examples
    """
    if type(one_hot) is not np.ndarray:
        return None
    if one_hot is None:
        return None
    if len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    Y = []
    for i in range(len(one_hot[0])):
        for j in range(len(one_hot)):
            if one_hot[j][i] == 1:
                Y.append(j)
    return np.array(Y)
