#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """ Add two arrays """
    if len(arr1) != len(arr2):
        return None
    array = []
    for i, j in zip(arr1, arr2):
        array.append(i + j)
    return array
