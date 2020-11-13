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
    if P.sum(axis=1).all() != 1:
        return False
    # Diagonal
    diag = np.diagonal(P)
    # Absorbing states
    abs_states = np.where(diag == 1)[0]
    # No absorbing states
    nabs_sta = np.where(diag != 1)[0]
    if len(abs_states) == 0:
        return False
    if len(nabs_sta) == 0:
        return True
    while True:
        # Select states that could be transformed in an absorbing state
        tr = np.where(np.logical_and(P[:, abs_states] > 0,
                                     P[:, abs_states] < 1))[0]
        tr = np.unique(tr)
        # If any value could be transformed just return False
        if len(tr) == 0:
            return False
        c = np.copy(nabs_sta)
        # Check for the last no abs if it could be transformed
        if len(nabs_sta) == 1 and nabs_sta in tr:
            return True
        # Remove states that could be transformed to absorbing state
        cust_where = np.array([x for x in nabs_sta for j in tr if x == j])
        nabs_sta = np.delete(nabs_sta, np.where(nabs_sta == cust_where))
        if np.array_equal(nabs_sta, c):
            return False
        if len(nabs_sta) == 0:
            return True
        # Add the non-absorbing that could be transformed
        abs_states = np.unique(np.append(abs_states, tr))
