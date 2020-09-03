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
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    # Kernel dimension
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # Output image dimensions
    nh = (ih - kh) + 1
    nw = (iw - kw) + 1
    new_img = np.zeros((m, nh, nw))
    op_kernel = np.ndarray(kernel.shape)
    for n in range(m):
        x = y = 0
        img = images[n]
        while y < img.shape[0] - 2:
            op_kernel = np.sum(img[y:y+3, x:x+3] * kernel)
            new_img[n][y][x] = op_kernel
            if x + 1 == img.shape[0] - 2:
                x = 0
                y += 1
            else:
                x += 1
    return new_img
