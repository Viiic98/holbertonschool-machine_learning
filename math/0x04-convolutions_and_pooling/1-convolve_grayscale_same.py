#!/usr/bin/env python3
""" Convolution on gray scale 'SAME'"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a same convolution on grayscale images

        @images: is a numpy.ndarray with shape (m, h, w)
                 containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        @kernel: is a numpy.ndarray with shape (kh, kw)
                 containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        if necessary, the image should be padded with 0’s
        You are only allowed to use two for loops; any other
        loops of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    # Kernel dimension
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # Output dimensions
    nh = ih + 2 * int(kh / 2) - (kh - 1)
    nw = iw + 2 * int(kw / 2) - (kw - 1)
    new_img = np.zeros((m, nh, nw))
    # Padding all images
    pad_images = np.pad(images, ((0,), (1,), (1,)), 'constant')
    x = y = 0
    while y < nh:
        op_kernel = pad_images[:, y:y+kh, x:x+kw] * kernel
        new_img[:, y, x] = np.sum(np.sum(op_kernel, axis=1), axis=1)
        if x + 1 == nw:
            x = 0
            y += 1
        else:
            x += 1
    return new_img
