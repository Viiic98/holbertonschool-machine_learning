#!/usr/bin/env python3
""" Cofactor matrix """


def determinant(matrix):
    """ calculates the determinant of a matrix

        - matrix is a list of lists whose determinant
          should be calculated
        - If matrix is not a list of lists, raise a TypeError
          with the message matrix must be a list of lists
        - If matrix is not square, raise a ValueError with the
          message matrix must be a square matrix
        - The list [[]] represents a 0x0 matrix
        Returns: the determinant of matrix
    """
    if len(matrix[0]) == 0:
        return 1
    if len(matrix[0]) == 1:
        return matrix[0][0]
    det = 0
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1]
        det -= matrix[0][1] * matrix[1][0]
        return det
    sign = 1
    for j in range(len(matrix[0])):
        tmp = matrix[1:]
        for i in range(len(tmp)):
            tmp[i] = tmp[i][0:j] + tmp[i][j + 1:]
        det += (sign * matrix[0][j]) * determinant(tmp)
        sign *= -1
    return det


def cofactor(matrix):
    """ calculates the cofactor matrix of a matrix

        - matrix is a list of lists whose cofactor matrix should
          be calculated
        - If matrix is not a list of lists, raise a TypeError with
          the message matrix must be a list of lists
        - If matrix is not square or is empty, raise a ValueError
          with the message matrix must be a non-empty square matrix
        Returns: the cofactor matrix of matrix
    """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    for x in matrix:
        if type(x) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(x):
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
                    minor_matrix[i][j] = ((-1)**(i + j)) * determinant(tmp)
                else:
                    a = tmp[0][0]
                    b = tmp[0][1]
                    c = tmp[1][0]
                    d = tmp[1][1]
                    minor_matrix[i][j] = ((-1)**(i + j)) * (a * d - b * c)
            else:
                minor_matrix[i][j] = ((-1)**(i + j)) * tmp[0][0]
    return minor_matrix
