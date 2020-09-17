#!/usr/bin/env python3
""" Projection block with Keras """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block

    @A_prev: is the output from the previous layer
    @filters: is a tuple or list containing F11, F3, F12, respectively:
        - F11: is the number of filters in the first 1x1 convolution
        - F3: is the number of filters in the 3x3 convolution
        - F12: is the number of filters in the second 1x1 convolution
               as well as the 1x1 convolution in the shortcut connection
    @s: is the stride of the first convolution in both the main path and
        the shortcut connection
    - All convolutions inside the block should be followed by batch
      normalization along the channels axis and a rectified linear
      activation (ReLU), respectively.
    - All weights should use he normal initialization
    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    layer_0 = K.layers.Conv2D(F11, (1, 1), padding='same',
                              strides=s,
                              activation='relu')(A_prev)
    batch = K.layers.BatchNormalization()(layer_0)
    act = K.layers.Activation(K.activations.relu)(batch)
    layer_1 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(act)
    batch = K.layers.BatchNormalization()(layer_1)
    act = K.layers.Activation(K.activations.relu)(batch)
    layer_2 = K.layers.Conv2D(F12, (1, 1), padding='same',
                              activation='relu')(act)
    batch = K.layers.BatchNormalization()(layer_2)
    layer_3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                              strides=s,
                              activation='relu')(A_prev)
    batch_short = K.layers.BatchNormalization()(layer_3)
    add = K.layers.Add()([batch, batch_short])
    act = K.layers.Activation(K.activations.relu)(add)

    return act
