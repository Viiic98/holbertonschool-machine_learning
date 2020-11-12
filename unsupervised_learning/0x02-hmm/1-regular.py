#!/usr/bin/env python3
""" Regular """
import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain

        - P is a is a square 2D numpy.ndarray of shape (n, n) representing
          the transition matrix
            - P[i, j] is the probability of transitioning from state i to
              state j
            - n is the number of states in the markov chain
        Returns: a numpy.ndarray of shape (1, n) containing the steady state
                 probabilities, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2 or len(P) != len(P[0]):
        return None
    p_c = np.copy(P)
    for i in range(1000):
        p_c = np.matmul(p_c, P)
        if p_c.all() > 0:
            break
    if i == 999:
        return None
    # note transpose of P to find left eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # find index of eigenvalue = 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    # remember to normalize eigenvector to get a probability distribution
    return w / np.sum(w)
