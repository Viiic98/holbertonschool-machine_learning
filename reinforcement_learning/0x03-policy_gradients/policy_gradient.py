#!/usr/bin/env python3
""" Policy gradient """
import numpy as np


def policy(matrix, weight):
    """ computes to policy with a weight of a matrix """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / exp.sum()


def policy_gradient(state, weight):
    """ computes the Monte-Carlo policy gradient based on a
        state and a weight matrix

        state: matrix representing the current observation of the environment
        weight: matrix of random weight
        Return: the action and the gradient (in this order)
    """
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])
    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    softmax = softmax[action, :]
    dlog = softmax / P[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
