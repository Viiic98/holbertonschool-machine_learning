#!/usr/bin/env python3
""" Placeholber with TensorFlow """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ Placeholder with Tf """
    x = tf.placeholder("float", [None, nx], 'x')
    y = tf.placeholder("float", [None, classes], 'y')
    return x, y
