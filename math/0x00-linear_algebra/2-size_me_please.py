#!/usr/bin/env python3


def matrix_shape(matrix):
    m = []
    while type(matrix) is list:
        m.append(len(matrix))
        matrix = matrix[0]
    return m
