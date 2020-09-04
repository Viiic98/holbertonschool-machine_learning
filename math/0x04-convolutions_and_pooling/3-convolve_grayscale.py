#!/usr/bin/env python3
""" Convlution with stride """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ performs a convolution on grayscale images

        @images: is a numpy.ndarray with shape (m, h, w)
                 containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        @kernel: is a numpy.ndarray with shape (kh, kw)
                 containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        @padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
                ph is the padding for the height of the image
                pw is the padding for the width of the image
            the image should be padded with 0’s
        @stride: is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        You are only allowed to use two for loops; any other loops
        of any kind are not allowed Hint: loop over i and j
        Returns: a numpy.ndarray containing the convolved images
    """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    # Kernel dimensions
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]
        nh = (ih - kh) + (2 * ph) + 1
        nw = (iw - kw) + (2 * pw) + 1
    elif padding == 'same':
        ph = int(kh / 2)
        pw = int(kw / 2)
        nh = int(ih / sh)
        nw = int(iw / sw)
    elif padding == 'valid':
        ph = 0
        pw = 0
        nh = int(((ih - kh) / sh) + 1)
        nw = int(((iw - kw) / sw) + 1)
    new_img = np.zeros((m, nh, nw))
    pad_images = np.pad(images, ((0,), (ph,), (pw,)), 'constant')
    x = y = 0
    i = j = 0
    while j < nh:
        op_kernel = pad_images[:, y:y+kh, x:x+kw] * kernel
        new_img[:, j, i] = np.sum(np.sum(op_kernel, axis=1), axis=1)
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    return new_img
