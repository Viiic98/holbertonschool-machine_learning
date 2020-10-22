#!/usr/bin/env python3
""" Definiteness """
import numpy as np


def definiteness(matrix):
    """ calculates the definiteness of a matrix

        - matrix is a numpy.ndarray of shape (n, n) whose definiteness
          should be calculated
        - If matrix is not a numpy.ndarray, raise a TypeError with
          the message matrix must be a numpy.ndarray
        - If matrix is not a valid matrix, return None
        Return: the string Positive definite, Positive semi-definite,
                Negative semi-definite, Negative definite, or Indefinite if
                the matrix is positive definite, positive semi-definite,
                negative semi-definite, negative definite of indefinite,
                respectively
        - If matrix does not fit any of the above categories, return None
    """
    """if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")"""

    w, v = np.linalg.eig(matrix)
    # count to classify
    pos = 0
    sem_pos = 0
    neg = 0
    sem_neg = 0
    for x in w:
        if x > 0:
            pos += 1
        if x >= 0:
            sem_pos += 1
        if x < 0:
            neg += 1
        if x <= 0:
            sem_neg += 1
    # All eigenvalues are > 0
    if pos == len(w):
        return "Positive definite"
    # All eigenvalues are >= 0
    if sem_pos == len(w):
        return "Positive semi-definite"
    # All eigenvalues are < 0
    if neg == len(w):
        return "Negative definite"
    # All eigenvalues are <= 0
    if sem_neg == len(w):
        return "Negative semi-definite"
    if pos and neg:
        return "Indefinite"
    else:
        return None
