#!/usr/bin/env python3
""" Likelihood for Binomial distribution """
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
    # P(B | A)
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
