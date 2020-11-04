#!/usr/bin/env python3
""" K-means """
import numpy as np


def closest_centroid(X, centroids):
    """ returns an array containing the index to the nearest
        centroid for each point"""
    distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset

        - X is a numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k is a positive integer containing the number of clusters
        - iterations is a positive integer containing the maximum number of
          iterations that should be performed
        - If no change in the cluster centroids occurs between iterations,
          your function should return
        - Initialize the cluster centroids using a multivariate uniform
          distribution (based on0-initialize.py)
        - If a cluster contains no data points during the update step,
          reinitialize its centroid
        Returns: C, clss, or None, None on failure
            - C is a numpy.ndarray of shape (k, d) containing the centroid
              means for each cluster
            - clss is a numpy.ndarray of shape (n,) containing the index of the
              cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k < 1:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    centroids = np.random.uniform(low=min, high=max, size=(k, d))
    clss = closest_centroid(X, centroids)
    for i in range(iterations):
        copy = np.copy(centroids)
        for c in range(k):
            idx = np.where(clss == c)
            if len(idx[0]) == 0:
                centroids[c] = np.random.uniform(min, max, (1, d))
            else:
                mean = X[idx].mean(axis=0)
                centroids[c] = mean
        clss = closest_centroid(X, centroids)
        if np.array_equal(copy, centroids):
            break
    return centroids, clss
