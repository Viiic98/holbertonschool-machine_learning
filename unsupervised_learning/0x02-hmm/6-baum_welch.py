#!/usr/bin/env python3
""" Baum Welch """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model """
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])
    for t in range(1, T):
        F[:, t] = np.multiply(Emission[:, Observation[t]],
                              np.dot(Transition.T, F[:, t - 1]))
    P = F[:, T - 1].sum()
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model """
    T = Observation.shape[0]
    N, M = Emission.shape
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        beta[:, t] = np.dot(Transition,
                            np.multiply(Emission[:, Observation[t + 1]],
                                        beta[:, t + 1]))
    P = np.dot(Initial.T, np.multiply(Emission[:, Observation[0]], beta[:, 0]))
    return P, beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ performs the Baum-Welch algorithm for a hidden markov model

        - Observations is a numpy.ndarray of shape (T,) that contains the
          index of the observation
            - T is the number of observations
        - Transition is a numpy.ndarray of shape (M, M) that contains
          the initialized transition probabilities
            - M is the number of hidden states
        - Emission is a numpy.ndarray of shape (M, N) that contains the
          initialized emission probabilities
            - N is the number of output states
        - Initial is a numpy.ndarray of shape (M, 1) that contains the
          initialized starting probabilities
        - iterations is the number of times expectation-maximization
          should be performed
        Returns: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    T = Observations.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    M, N = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (M, M):
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    for n in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T, Transition) *
                                 Emission[:, Observations[t + 1]].T,
                                 beta[:, t + 1])
            for i in range(M):
                numerator = alpha[i, t] * Transition[i, :]\
                            * Emission[:, Observations[t + 1]].T *\
                            beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for i in range(N):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
