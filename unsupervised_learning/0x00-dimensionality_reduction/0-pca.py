#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset

        - X is a numpy.ndarray of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
            - all dimensions have a mean of 0 across all data points
        - var is the fraction of the variance that the PCA
          transformation should maintain
        Returns: the weights matrix, W, that maintains var fraction
                 of X‘s original variance
        - W is a numpy.ndarray of shape (d, nd) where nd is the new
          dimensionality of the transformed X
    """
    u, s, vh = np.linalg.svd(X)
    # sort in descending order eigenvalues
    s = s[::-1]
    # accumulative sum of eigenvalues
    cum = np.cumsum(s)
    # Get indices less than variance
    idx = [i for i in range(len(cum)) if cum[i] < var]
    # vh Transpose is equal to W
    return vh.T[:, idx]
