#!/usr/bin/env python3
""" Training using TensorFlow """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ creates the training operation for the network

        @loss: is the loss of the network’s prediction
        @alpha is the learning rate

        Return: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
