#!/usr/bin/env python3
""" Transition layer with Keras """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer

    @X: is the output from the previous layer
    @nb_filters: is an integer representing the number of filters in X
    @compression: is the compression factor for the transition layer
    - Your code should implement compression as used in DenseNet-C
    - All weights should use he normal initialization
    - All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of
    filters within the output, respectively
    """
    init = K.initializers.he_normal()
    batch = K.layers.BatchNormalization()(X)
    act = K.layers.Activation(K.activations.relu)(batch)
    conv = K.layers.Conv2D(int(nb_filters * compression), (1, 1),
                           padding='same',
                           strides=1,
                           kernel_initializer=init,
                           activation='relu')(act)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=2)(conv)
    return avg_pool, int(nb_filters * compression)
