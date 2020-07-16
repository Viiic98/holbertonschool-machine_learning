#!/usr/bin/env python3
""" Concatenate Numpy Array"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
        Function that concatenates numpy arrays
    """
    return np.concatenate((mat1, mat2), axis=axis)
