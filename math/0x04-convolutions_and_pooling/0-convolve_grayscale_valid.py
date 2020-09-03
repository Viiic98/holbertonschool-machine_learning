#!/usr/bin/env python3
""" Convolution on grayscale """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale images

        @images: is a numpy.ndarray with shape (m, h, w)
                 containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        @kernel: is a numpy.ndarray with shape (kh, kw)
                 containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        You are only allowed to use two for loops; any other
        loops of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    new_img = np.ndarray((m, h - 2, w - 2))
    op_kernel = np.ndarray(kernel.shape)
    for n in range(m):
        x = y = 0
        while (x and y) < h - 2:
            op_kernel = np.sum(images[n][x:x+3, y:y+3] * kernel)
            new_img[n][x][y] = op_kernel
            if x + 1 == h - 2:
                y += 1
                if y < 26:
                    x = 0
            x += 1
    return new_img
