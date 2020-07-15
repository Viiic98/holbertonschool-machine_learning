#!/usr/bin/env python3
""" Calculate shape of a matrix """


def matrix_shape(matrix):
    """ Calculate shape of a matrix """
    m = []
    while type(matrix) is list:
        m.append(len(matrix))
        if matrix[0]:
            matrix = matrix[0]
        else:
            break
    return m
