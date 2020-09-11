#!/usr/bin/env python3
""" Convolution of a layer """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """ performs forward propagation over a convolutional
        layer of a neural network

        @A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                 containing the output of the previous layer
            - m: is the number of examples
            - h_prev: is the height of the previous layer
            - w_prev: is the width of the previous layer
            - c_prev: is the number of channels in the previous layer
        @W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
            - kh: is the filter height
            - kw: is the filter width
            - c_prev: is the number of channels in the previous layer
            - c_new: is the number of channels in the output
        @b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        @activation: is an activation function applied to the convolution
        @padding: is a string that is either same or valid, indicating the type
                  of padding used
        @stride: is a tuple of (sh, sw) containing the strides for the
                 convolution
            - sh: is the stride for the height
            - sw: is the stride for the width
        you may import numpy as np
        Returns: the output of the convolutional layer
    """
    # Define variables
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    # Set padding
    if type(padding) == tuple:
        ph, pw = padding.shape
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh - h_prev + kh) / 2)) + 1
        pw = int((((w_prev - 1) * sw - w_prev + kw) / 2)) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Output dimensions
    nh = int(((h_prev + (2 * ph) - kh) / sh)) + 1
    nw = int(((w_prev + (2 * pw) - kw) / sw)) + 1

    pad_images = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), 'constant')
    new_img = np.zeros((m, nh, nw, c_new))
    x = y = 0
    i = j = 0
    while j < nh:
        k = 0
        while k < c_new:
            op_filter = (pad_images[:, y:y+kh, x:x+kw, :]
                         * W[:, :, :, k])
            new_img[:, j, i, k] = activation(op_filter.sum(axis=(1, 2, 3)))
            k += 1
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    new_img += b
    return new_img
