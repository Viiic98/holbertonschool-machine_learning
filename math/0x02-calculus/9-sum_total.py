#!/usr/bin/env python3
""" Sum square with recursion """


def summation_i_squared(n):
    """ Recursive function """
    if (type(n) != int and type(n) != float):
        return None
    if (n < 1):
        return None
    return sum(map(lambda i: i*i, range(n + 1)))
