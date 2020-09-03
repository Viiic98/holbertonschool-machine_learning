#!/usr/bin/env python3
""" Convlution with padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a convolution on grayscale images with
        custom padding

        @images: is a numpy.ndarray with shape (m, h, w)
                 containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        @kernel: is a numpy.ndarray with shape (kh, kw)
                 containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        @padding: is a tuple of (ph, pw)
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image
        the image should be padded with 0’s
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
    # padding dimensions
    ph = padding[0]
    pw = padding[1]
    # new image dimensions
    nh = (ih - kh) + (2 * ph) + 1
    nw = (iw - kw) + (2 * pw) + 1
    new_img = np.ndarray((m, nh, nw))
    op_kernel = np.ndarray(kernel.shape)
    for n in range(m):
        x = y = 0
        img = np.pad(images[n], ((ph,), (pw,)), 'constant')
        while y < img.shape[0] - 2:
            op_kernel = np.sum(img[y:y+3, x:x+3] * kernel)
            new_img[n][y][x] = op_kernel
            if x + 1 == img.shape[0] - 2:
                x = 0
                y += 1
            else:
                x += 1
    return new_img
