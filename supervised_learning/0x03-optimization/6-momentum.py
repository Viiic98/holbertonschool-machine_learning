#!/usr/bin/env python3
""" Applying gradient descent with momentum TF """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates the training operation for a neural network
        in tensorflow using the gradient descent with momentum
        optimization algorithm

        @loss: is the loss of the network
        @alpha: is the learning rate
        @beta1: is the momentum weight
        Returns: the momentum optimization operation
    """
    optmizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optmizer.minimize(loss)
    return train
