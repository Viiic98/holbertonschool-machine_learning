#!/usr/bin/env python3
""" Bayesian Optimization """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ Bayesian optimization on a noiseless 1D Gaussian process """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Class contructor

            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
              already sampled with the black-box function
            - Y_init is a numpy.ndarray of shape (t, 1) representing the
              outputs of the black-box function for each input in X_init
            - t is the number of initial samples
            - bounds is a tuple of (min, max) representing the bounds of the
              space in which to look for the optimal point
            - ac_samples is the number of samples that should be analyzed
              during acquisition
            - l is the length parameter for the kernel
            - sigma_f is the standard deviation given to the output of the
              black-box function
            - xsi is the exploration-exploitation factor for acquisition
            - minimize is a bool determining whether optimization should be
              performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ calculates the next best sample location

            - Uses the Expected Improvement acquisition function
            Returns: X_next, EI
                - X_next is a numpy.ndarray of shape (1,) representing the next
                  best sample point
                - EI is a numpy.ndarray of shape (ac_samples,) containing the
                  expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        if not self.minimize:
            X_next = np.amax(self.gp.Y)
            imp = (mu - X_next - self.xsi)
        if self.minimize:
            X_next = np.amin(self.gp.Y)
            imp = (X_next - mu - self.xsi)
        n = len(sigma)
        Z = [imp[i] / sigma[i] if sigma[i] > 0 else 0 for i in range(n)]
        EI = np.zeros(sigma.shape)
        for i in range(n):
            if sigma[i] > 0:
                EI[i] = imp[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """ optimizes the black-box function

            - iterations is the maximum number of iterations to perform
            - If the next proposed point is one that has already been sampled,
              optimization should be stopped early
            Returns: X_opt, Y_opt
                - X_opt is a numpy.ndarray of shape (1,) representing the
                  optimal point
                - Y_opt is a numpy.ndarray of shape (1,) representing the
                  optimal function value
        """
        sampled = []
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in sampled:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            sampled.append(X_next)

        if not self.minimize:
            idx = np.argmax(self.gp.Y)
        if self.minimize:
            idx = np.argmin(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]

    @staticmethod
    def f(x):
        """our 'black box' function"""
        return np.sin(5 * x) + 2 * np.sin(-2 * x)
