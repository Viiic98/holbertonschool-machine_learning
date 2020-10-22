#!/usr/bin/env python3
""" Minor Matrix """
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """ calculates the minor matrix of a matrix

        - matrix is a list of lists whose minor matrix
          should be calculated
        - If matrix is not a list of lists, raise a TypeError
          with the message matrix must be a list of lists
        - If matrix is not square or is empty, raise a ValueError
          with the message matrix must be a non-empty square matrix
        Returns: the minor matrix of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    if type(matrix) is list and len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if type(matrix) is list and len(matrix) > 0:
        if type(matrix[0]) is not list:
            raise TypeError("matrix must be a list of lists")
    if len(matrix) > 0 and len(matrix[0]) > 0:
        if len(matrix) != len(matrix[0]):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]
    minor_matrix = [x[:] for x in matrix]
    for i in range(len(matrix)):
        sub = matrix[:i] + matrix[i + 1:]
        for j in range(len(matrix)):
            tmp = sub[:]
            for k in range(len(tmp)):
                tmp[k] = tmp[k][0:j] + tmp[k][j + 1:]
            if len(tmp) > 1:
                if len(tmp) > 2:
                    minor(tm)
                a = tmp[0][0]
                b = tmp[0][1]
                c = tmp[1][0]
                d = tmp[1][1]
                minor_matrix[i][j] = a * d - b * c
            else:
                minor_matrix[i][j] = tmp[0][0]
    return minor_matrix
