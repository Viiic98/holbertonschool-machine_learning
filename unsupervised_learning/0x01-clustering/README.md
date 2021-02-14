# Clustering

![c](https://media.geeksforgeeks.org/wp-content/uploads/merge3cluster.jpg)

## Requeriments
- Unless otherwise noted, you are not allowed to import any module except import numpy as np
- Your code should use the minimum number of operations

## Tasks

### [Initialize K-means](./0-initialize.py)
- Write a function def initialize(X, k): that initializes cluster centroids for K-means:
    - You should use numpy.random.uniform exactly once
    - You are not allowed to use any loops

### [K-means](./1-kmeans.py)
- Write a function def kmeans(X, k, iterations=1000): that performs K-means on a dataset:
    - You should use numpy.random.uniform exactly twice
    - You may use at most 2 loops

### [Variance](./2-variance.py)
- Write a function def variance(X, C): that calculates the total intra-cluster variance for a data set:
    - You are not allowed to use any loops

### [Optimize k](./3-optimum.py)
- Write a functiondef optimum_k(X, kmin=1, kmax=None, iterations=1000): that tests for the optimum number of clusters by variance:
    - This function should analyze at least 2 different cluster sizes
    - You should use:
        - kmeans = __import__('1-kmeans').kmeans
        - variance = __import__('2-variance').variance
    - You may use at most 2 loops 

### [Initialize GMM](./4-initialize.py)
- Write a function def initialize(X, k): that initializes variables for a Gaussian Mixture Model:
    - You are not allowed to use any loops
    - You should use kmeans = __import__('1-kmeans').kmeans

### [PDF](./5-pdf.py)
- Write a function def pdf(X, m, S): that calculates the probability density function of a Gaussian distribution:
    - You are not allowed to use any loops
    - You are not allowed to use the function numpy.diag or the method numpy.ndarray.diagonal

### [Expectation](./6-expectation.py)
- Write a function def expectation(X, pi, m, S): that calculates the expectation step in the EM algorithm for a GMM:
    - You may use at most 1 loop
    - You should use pdf = __import__('5-pdf').pdf

### [Maximization](./7-maximization.py)
- Write a function def maximization(X, g): that calculates the maximization step in the EM algorithm for a GMM:
    - You may use at most 1 loop

### [EM](./8-EM.py)
- Write a function def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False): that performs the expectation maximization for a GMM:
    - You should use:
        - initialize = __import__('4-initialize').initialize
        - expectation = __import__('6-expectation').expectation
        - maximization = __import__('7-maximization').maximization
    - You may use at most 1 loop

### [BIC](./9-BIC.py)
- Write a function def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False): that finds the best number of clusters for a GMM using the Bayesian Information Criterion:
    - You should use expectation_maximization = __import__('7-EM').expectation_maximization
    - You may use at most 1 loop

### [Hello, sklearn!](./10-kmeans.py)
- Write a function def kmeans(X, k): that performs K-means on a dataset:
    - The only import you are allowed to use is import sklearn.cluster

### [GMM](./11-gmm.py)
- Write a function def gmm(X, k): that calculates a GMM from a dataset:
    - The only import you are allowed to use is import sklearn.mixture

### [Agglomerative](./12-agglomerative.py)
- Write a function def agglomerative(X, dist): that performs agglomerative clustering on a dataset:
    - The only imports you are allowed to use are:
        - import scipy.cluster.hierarchy
        - import matplotlib.pyplot as plt
