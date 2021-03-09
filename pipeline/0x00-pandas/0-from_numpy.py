#!/usr/bin/env python3
""" From numpy """
import pandas as pd


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray

        - array is the np.ndarray from which you should create the pd.DataFrame
        - The columns of the pd.DataFrame should be labeled in alphabetical
          order
          and capitalized. There will not be more than 26 columns.
        Returns: the newly created pd.DataFrame
    """
    cols = list(map(chr, range(65, 65 + len(array[0]))))
    return pd.DataFrame(array, columns=[cols])
