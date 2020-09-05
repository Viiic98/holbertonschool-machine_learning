#!/usr/bin/env python3
""" Convolution with multiple kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ that performs a convolution on images using multiple kernels

        @images: is a numpy.ndarray with shape (m, h, w, c)
                 containing multiple images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        @kernels: is a numpy.ndarray with shape (kh, kw, c, nc)
                  containing the kernels for the convolution
            - kh is the height of a kernel
            - kw is the width of a kernel
            - nc is the number of kernels
        @padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
            - if ‘same’, performs a same convolution
            - if ‘valid’, performs a valid convolution
            - if a tuple:
                - ph is the padding for the height of the image
                - pw is the padding for the width of the image
            - the image should be padded with 0’s
        @stride: is a tuple of (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
        You are only allowed to use three for loops; any other loops
        of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    c = images.shape[3]
    # Kernel dimensions
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        ph = int((((ih - 1) * sh - ih + kh) / 2)) + 1
        pw = int((((iw - 1) * sw - iw + kw) / 2)) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    pad_images = np.pad(images, ((0,), (ph,), (pw,), (0,)), 'constant')
    nh = int(((ih + (2 * ph) - kh) / sh)) + 1
    nw = int(((iw + (2 * pw) - kw) / sw)) + 1
    new_img = np.zeros((m, nh, nw, nc))
    x = y = 0
    i = j = 0
    while j < nh:
        k = 0
        while k < nc:
            op_kernel = (pad_images[:, y:y+kh, x:x+kw, :]
                         * kernels[:, :, :, k])
            new_img[:, j, i, k] = op_kernel.sum(axis=(1, 2, 3))
            k += 1
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    return new_img
