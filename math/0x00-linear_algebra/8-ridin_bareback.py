#!/usr/bin/env python3
""" Matrix multiplication """


def mat_mul(mat1, mat2):
    """ Dot product
        This can be only applied if the number of columns
        of the first matrix are equal to the number of
        rows in the second matrix
    """
    if mat1 == [] or mat2 == []:
        return None
    if len(mat1[0]) != len(mat2):
        return None
    if len(mat1) >= len(mat2):
        rows = len(mat1)
    else:
        rows = len(mat2)
    if len(mat1[0]) >= len(mat2[0]):
        colums = len(mat1[0])
    else:
        colums = len(mat2[0])
    matrix = [[0] * colums for _ in range(rows)]
    x = 0
    y = 0
    for i in mat1:
        for j, k in zip(i, mat2):
            y = 0
            for l in k:
                matrix[x][y] += j * l
                y += 1
        x += 1
    return matrix
