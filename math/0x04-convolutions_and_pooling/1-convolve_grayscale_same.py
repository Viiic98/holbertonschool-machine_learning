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
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    new_img = np.ndarray((m, h, w))
    op_kernel = np.ndarray(kernel.shape)
    for n in range(m):
        x = y = 0
        img = np.pad(images[n], 1, 'constant')
        while y < img.shape[0] - 2:
            op_kernel = np.sum(img[y:y+3, x:x+3] * kernel)
            new_img[n][y][x] = op_kernel
            if x + 1 == img.shape[0] - 2:
                x = 0
                y += 1
            else:
                x += 1
    return new_img
