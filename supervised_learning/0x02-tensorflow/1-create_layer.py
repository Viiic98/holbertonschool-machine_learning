#!/usr/bin/env python3
""" Layer creation """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Create a layer and return output tensor of layer

        @prev: is the tensor output of the previous layer
        @n: is the number of nodes in the layer to create
        @activation: is the activation function that the layer should use
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation, name='layer', kernel_initializer=w)
    out = layer(prev)
    return out
