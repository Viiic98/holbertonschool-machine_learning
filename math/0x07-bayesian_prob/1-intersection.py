#!/usr/bin/env python3
""" Intersection with prior """
import numpy as np


def likelihood(x, n, P):
    """ calculates the likelihood of obtaining this data given
        various hypothetical probabilities of developing severe
        side effects

        - x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of developing severe side effects
        - If n is not a positive integer, raise a ValueError with the
          message n must be a positive integer
        - If x is not an integer that is greater than or equal to 0,
          raise a ValueError with the message x must be an integer
          that is greater than or equal to 0
        - If x is greater than n, raise a ValueError with the message
          x cannot be greater than n
        - If P is not a 1D numpy.ndarray, raise a TypeError with the
          message P must be a 1D numpy.ndarray
        - If any value in P is not in the range [0, 1], raise a ValueError
          with the message All values in P must be in the range [0, 1]
        Returns: a 1D numpy.ndarray containing the likelihood of obtaining
                 the data, x and n, for each probability in P, respectively
    """
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         "equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if len(P.shape) != 1 or type(P) is not np.ndarray:
        raise TypeError("P must be a 1D numpy.ndarray")
    for i in P:
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    fact = np.math.factorial
    a = (fact(n)) / (fact(x) * fact(n - x))
    b = P**x * (1 - P)**(n - x)
    return a * b


def intersection(x, n, P, Pr):
    """ calculates the intersection of obtaining this data with the
        various hypothetical probabilities

        - x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of developing severe side effects
        - Pr is a 1D numpy.ndarray containing the prior beliefs of P
        - If n is not a positive integer, raise a ValueError with the
          message n must be a positive integer
        - If x is not an integer that is greater than or equal to 0,
          raise a ValueError with the message x must be an integer that is
          greater than or equal to 0
        - If x is greater than n, raise a ValueError with the message x
          cannot be greater than n
        - If P is not a 1D numpy.ndarray, raise a TypeError with the
          message P must be a 1D numpy.ndarray
        - If Pr is not a numpy.ndarray with the same shape as P, raise a
          TypeError with the message Pr must be a numpy.ndarray with the
          same shape as P
        - If any value in P or Pr is not in the range [0, 1], raise a
          ValueError with the message All values in {P} must be in the
          range [0, 1] where {P} is the incorrect variable
        - If Pr does not sum to 1, raise a ValueError with the message
          Pr must sum to 1 Hint: use numpy.isclose
        - All exceptions should be raised in the above order
        Returns: a 1D numpy.ndarray containing the intersection of obtaining
                 x and n with each probability in P, respectively
    """
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         "equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if len(P.shape) != 1 or type(P) is not np.ndarray:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i, j in zip(P, Pr):
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        if j < 0 or j > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    L = likelihood(x, n, P)
    return L * Pr
