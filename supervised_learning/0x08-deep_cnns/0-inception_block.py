#!/usr/bin/env python3
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ builds an inception block

    @A_prev: is the output from the previous layer
    @filters: is a tuple or list containing
              F1, F3R, F3,F5R, F5, FPP, respectively:
        - F1: is the number of filters in the 1x1 convolution
        - F3R: is the number of filters in the 1x1 convolution
               before the 3x3 convolution
        - F3: is the number of filters in the 3x3 convolution
        - F5R: is the number of filters in the 1x1 convolution
               before the 5x5 convolution
        - F5: is the number of filters in the 5x5 convolution
        - FPP: is the number of filters in the 1x1 convolution
               after the max pooling
    - All convolutions inside the inception block should use a
    rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    # Get number of filters for every layer
    F1, F3R, F3, F5R, F5, FPP = filters
    # Layer 1x1
    layer_0 = K.layers.Conv2D(F1, (1, 1), padding='same',
                              activation='relu')(A_prev)
    # Layer 1x1 then 3x3
    layer_1 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                              activation='relu')(A_prev)
    layer_1 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(layer_1)
    # layer 1x1 then 5x5
    layer_2 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                              activation='relu')(A_prev)
    layer_2 = K.layers.Conv2D(F5, (5, 5), padding='same',
                              activation='relu')(layer_2)
    # layer 3x3 then 1x1
    layer_3 = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                    padding='same')(A_prev)
    layer_3 = K.layers.Conv2D(FPP, (1, 1), padding='same',
                              activation='relu')(layer_3)
    # Concatenate previous layers
    block = K.layers.concatenate([layer_0, layer_1, layer_2, layer_3],
                                 axis=3)
    return block
