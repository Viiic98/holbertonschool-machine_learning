#!/usr/bin/env python3
""" Presicion """
import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a
        confusion matrix

        @confusion: is a confusion numpy.ndarray of shape
                    (classes, classes) where row indices represent
                    the correct labels and column indices represent
                    the predicted labels
        @classes: is the number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the
                 precision of each class
    """
    prec = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        tp = confusion[i][i]
        fp = np.sum(confusion, axis=0) - tp
        prec[i] = tp / (tp + fp[i])
    return prec
