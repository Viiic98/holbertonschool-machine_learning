#!/usr/bin/env python3
""" Batch normalization with tensorflow """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ creates a batch normalization layer for a neural network in tensorflow

        @prev: is the activated output of the previous layer
        @n: is the number of nodes in the layer to be created
        @activation: is the activation function that should be used on the
                     output of the layer
        - you should use the tf.layers.Dense layer as the base layer with
          kernal initializer tf.contrib.layers.variance_scaling_initializer
          (mode="FAN_AVG")
        - your layer should incorporate two trainable parameters, gamma and
          beta, initialized as vectors of 1 and 0 respectively
        - you should use an epsilon of 1e-8
        Returns: a tensor of the activated output for the layer
    """
    # Kernel initializer
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    # Create the layer
    layer = tf.layers.Dense(units=n, kernel_initializer=w, name='layer')
    # Add input to layer
    x = layer(prev)
    # Mean and variance
    mean, variance = tf.nn.moments(x, axes=[0])
    # Hyperparameters gamma and beta
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    # Implement batch normalization
    norm = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    # Activate the output
    act = activation(norm)
    return act
