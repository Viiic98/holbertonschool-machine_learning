#!/usr/bin/env python3
""" Transpose a matrix """
matrix_shape = __import__('2-size_me_please').matrix_shape


def matrix_transpose(matrix):
    """ Transpose a matrix """
    shape = matrix_shape(matrix)
    t_matrix = [[0]*shape[0] for _ in range(shape[1])]
    l = 0
    for i in range(len(matrix)):
        k = 0
        for j in range(len(matrix[i])):
            t_matrix[k][l] = matrix[i][j]
            k += 1
        l += 1
    return t_matrix
