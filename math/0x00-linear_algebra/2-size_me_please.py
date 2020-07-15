#!/usr/bin/env python3


def matrix_shape(matrix):
    m = []
    while type(matrix) is list:
        m.append(len(matrix))
        if matrix[0]:
            matrix = matrix[0]
        else:
            break
    return m
