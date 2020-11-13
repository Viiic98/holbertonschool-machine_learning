#!/usr/bin/env python3
""" Absorbing """
import numpy as np


def absorbing(P):
    """ determines if a markov chain is absorbing

        - P is a is a square 2D numpy.ndarray of shape (n, n) representing
          the standard transition matrix
            - P[i, j] is the probability of transitioning from state i to
              state j
            - n is the number of states in the markov chain
        Returns: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2 or len(P) != len(P[0]):
        return False
    diag = np.diagonal(P)
    abs_states = np.where(diag == 1)[0]
    nabs_sta = np.where(diag != 1)[0]
    if len(abs_states) == 0:
        return False
    if len(nabs_sta) == 0:
        return True
    while True:
        tr = np.where(np.logical_and(P[:, abs_states] > 0,
                                     P[:, abs_states] < 1))[0]
        tr = np.unique(tr)
        if len(tr) == 0:
            return False
        c = np.copy(nabs_sta)
        if len(nabs_sta) == 1 and nabs_sta in tr:
            return True
        nabs_sta = np.delete(nabs_sta, np.where(nabs_sta == tr)[0])
        if np.array_equal(nabs_sta, c):
            return False
        if len(nabs_sta) == 0:
            return True
        abs_states = np.unique(np.append(abs_states, tr))
