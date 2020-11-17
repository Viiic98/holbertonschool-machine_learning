#!/usr/bin/env python3
""" Backward """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model

        - Observation is a numpy.ndarray of shape (T,) that contains
          the index of the observation
            - T is the number of observations
        - Emission is a numpy.ndarray of shape (N, M) containing the
          emission probability of a specific observation given a hidden state
            - Emission[i, j] is the probability of observing j given the
              hidden state i
            - N is the number of hidden states
            - M is the number of all possible observations
        - Transition is a 2D numpy.ndarray of shape (N, N) containing the
          transition probabilities
            - Transition[i, j] is the probability of transitioning from the
              hidden state i to j
        - Initial a numpy.ndarray of shape (N, 1) containing the probability
          of starting in a particular hidden state
        Returns: P, B, or None, None on failure
            - Pis the likelihood of the observations given the model
            - B is a numpy.ndarray of shape (N, T) containing the backward
              path probabilities
                - B[i, j] is the probability of generating the future
                  observations from hidden state i at time j
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        beta[:, t] = np.dot(Transition,
                            np.multiply(Emission[:, Observation[t + 1]],
                                        beta[:, t + 1]))
    P = np.dot(Initial.T, np.multiply(Emission[:, Observation[0]], beta[:, 0]))
    return P, beta
