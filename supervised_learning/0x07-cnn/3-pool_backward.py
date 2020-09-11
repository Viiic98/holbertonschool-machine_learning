#!/usr/bin/env python3
""" Pooling backward propagation """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1),
                  mode='max'):
    """ performs back propagation over a pooling layer of a neural
        network

        @dA: is a numpy.ndarray of shape (m, h_new, w_new, c_new)
             containing the partial derivatives with respect to the
             output of the pooling layer
            m: is the number of examples
            h_new: is the height of the output
            w_new: is the width of the output
            c: is the number of channels
        @A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c)
                 containing the output of the previous layer
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
        @kernel_shape: is a tuple of (kh, kw) containing the size of
                       the kernel for the pooling
            kh: is the kernel height
            kw: is the kernel width
        @stride: is a tuple of (sh, sw) containing the strides for
                 the pooling
            sh: is the stride for the height
            sw: is the stride for the width
        @mode: is a string containing either max or avg, indicating
               whether to perform maximum or average pooling, respectively
        you may import numpy as np
        Returns: the partial derivatives with respect to the previous layer
                 (dA_prev)
    """
    # dA
    m, h_new, w_new, c_new = dA.shape
    # A_prev
    m, h_prev, w_prev, c = A_prev.shape
    # Kernel
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    # Stride
    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Variables to define slice size
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        # Slice a_prev_pad
                        a_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, h_start:h_end,
                                w_start:w_end, c] += mask*dA[i, h, w, c]
                    if mode == 'avg':
                        average_dA = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i, h_start:h_end,
                                w_start:w_end, c] += np.ones((kh, kw))\
                            * average_dA
    return dA_prev
