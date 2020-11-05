#!/usr/bin/env python3
""" GMM PDF """
import numpy as np


def pdf(X, m, S):
    """ calculates the probability density function of a Gaussian distribution

        - X is a numpy.ndarray of shape (n, d) containing the data points whose
          PDF should be evaluated
        - m is a numpy.ndarray of shape (d,) containing the mean of the
          distribution
        - S is a numpy.ndarray of shape (d, d) containing the covariance of the
          distribution
        Returns: P, or None on failure
            - P is a numpy.ndarray of shape (n,) containing the PDF values for
              each data point
        - All values in P should have a minimum value of 1e-300
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    a = 1 / np.sqrt((((2 * np.pi) ** (d) * det)))
    inv = np.matmul((X - m), inv)
    np.seterr(over='ignore')
    b = np.exp(-(1 / 2) * ((np.matmul(inv, (X - m).T))))
    pdf = a * b
    pdf = np.where(pdf >= 1e-300, pdf, 1e-300)
    idx = np.where(np.eye(pdf.shape[0], dtype=bool))
    return pdf[idx]
