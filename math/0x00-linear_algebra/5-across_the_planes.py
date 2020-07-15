#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    shape = matrix_shape(mat1)
    matrix = []
    for axis1, axis2 in zip(mat1, mat2):
        matrix.append(add_arrays(axis1, axis2))
    return matrix
