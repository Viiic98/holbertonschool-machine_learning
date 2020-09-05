#!/usr/bin/env python3
""" Convolution with pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images

        @images: is a numpy.ndarray with shape (m, h, w, c)
                 containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        @kernel_shape: is a tuple of (kh, kw) containing the kernel
                       shape for the pooling
            kh is the height of the kernel
            kw is the width of the kernel
        @stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        @mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling
        You are only allowed to use two for loops; any other loops
        of any kind are not allowed
        Returns: a numpy.ndarray containing the pooled images
    """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    c = images.shape[3]
    # Kernel dimensions
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    nh = int(((ih - kh) / sh)) + 1
    nw = int(((iw - kw) / sw)) + 1
    new_img = np.zeros((m, nh, nw, c))
    x = y = 0
    i = j = 0
    while j < nh:
        if mode == 'max':
            op_kernel = np.max(images[:, y:y+kh, x:x+kw, :],
                               axis=(1, 2))
        elif mode == 'avg':
            op_kernel = np.average(images[:, y:y+kh, x:x+kw, :],
                                   axis=(1, 2))
        new_img[:, j, i] = op_kernel
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    return new_img
