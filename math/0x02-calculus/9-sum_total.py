#!/usr/bin/env python3
""" Sum square with recursion """


def summation_i_squared(n):
    """ Recursive function """
    if (n < 0):
        return None
    if (n >= 1):
        return(summation_i_squared(n - 1) + n**2)
    return (n**2)
