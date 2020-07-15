#!/usr/bin/env python3
""" Concatenates two matrices along a specific axis """
import copy
cat_arrays = __import__('6-howdy_partner').cat_arrays


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenate 2D matrix
        Axis 0 = Colum
        Axis 1 = Row
    """
    matrix = []
    if axis == 0:
        for axis in copy.deepcopy(mat1):
            matrix.append(axis)
        for axis in copy.deepcopy(mat2):
            matrix.append(axis)
    else:
        for i, j in zip(mat1, mat2):
            matrix.append(cat_arrays(i, j))
    return matrix
