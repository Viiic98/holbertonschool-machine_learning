#!/usr/bin/env python3
""" Concatenates two matrices along a specific axis """
cat_arrays = __import__('6-howdy_partner').cat_arrays


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenate 2D matrix
        Axis 0 = Colum
        Axis 1 = Row
    """
    matrix = []
    mat1_copy = [row[:] for row in mat1[:]]
    mat2_copy = [row[:] for row in mat2[:]]
    if axis == 0:
        for axis in mat1_copy:
            matrix.append(axis)
        for axis in mat2_copy:
            matrix.append(axis)
    else:
        for i, j in zip(mat1_copy, mat2_copy):
            matrix.append(cat_arrays(i, j))
    if len(matrix):
        return matrix
    return None
