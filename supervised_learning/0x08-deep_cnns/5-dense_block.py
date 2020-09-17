#!/usr/bin/env python3
""" Dense block with Keras """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block

    @X: is the output from the previous layer
    @nb_filters: is an integer representing the number of
                 filters in X
    @growth_rate: is the growth rate for the dense block
    @layers: is the number of layers in the dense block
    - You should use the bottleneck layers used for DenseNet-B
    - All weights should use he normal initialization
    - All convolutions should be preceded by Batch Normalization
     and a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the
             Dense Block and the number of filters within the
             concatenated outputs, respectively
    """
    init = K.initializers.he_normal()

    for i in range(layers):
        batch = K.layers.BatchNormalization()(X)
        act = K.layers.Activation(K.activations.relu)(batch)
        conv = K.layers.Conv2D(growth_rate * 4, (1, 1), padding='same',
                               strides=1,
                               kernel_initializer=init,
                               activation='relu')(act)
        batch = K.layers.BatchNormalization()(conv)
        act = K.layers.Activation(K.activations.relu)(batch)
        conv = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                               strides=1,
                               kernel_initializer=init,
                               activation='relu')(act)
        X = K.layers.concatenate([X, conv], axis=3)
        nb_filters += growth_rate
    return X, nb_filters
