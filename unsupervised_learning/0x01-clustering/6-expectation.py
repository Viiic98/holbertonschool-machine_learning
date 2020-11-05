#!/usr/bin/env python3
""" Expectation GMM """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM

        - X is a numpy.ndarray of shape (n, d) containing the data set
        - pi is a numpy.ndarray of shape (k,) containing the priors for each
          cluster
        - m is a numpy.ndarray of shape (k, d) containing the centroid means
          for each cluster
        - S is a numpy.ndarray of shape (k, d, d) containing the covariance
          matrices for each cluster
        Returns: g, l, or None, None on failure
            - g is a numpy.ndarray of shape (k, n) containing the posterior
              probabilities for each data point in each cluster
            - l is the total log likelihood
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if (pi < 0).all():
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    np.seterr(over='ignore')
    n, d = X.shape
    k = pi.shape[0]
    if (k, d) != m.shape or (k, d, d) != S.shape:
        return None, None
    g = []
    for i in range(k):
        P = pdf(X, m[i], S[i]) * pi[i]
        g.append(P)
    g = np.array(g)
    likelihood = np.log(g.sum(axis=0)).sum()
    g /= g.sum(axis=0)
    return g, likelihood
