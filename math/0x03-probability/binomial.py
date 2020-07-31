#!/usr/bin/env python3
""" Binomial class"""


class Binomial():
    """ Binomial distribution representation"""
    def __init__(self, data=None, n=1, p=0.5):
        """ Class constructor

            Data is a list of the data to be used to estimate the distribution
            n is the number of Bernoulli trials
            p is the probability of a “success”
        """
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.n = int(len(data) / 2)
                self.p = (sum(data) / self.n) / 100
        else:
            if n < 0:
                raise ValueError("n must be a positive value")
            elif p < 0 and p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
            self.n = int(n)

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of “successes”

            k is the number of “successes”
            q is the prob of failure
            PMF = (comb n, k) * (p**k) * q**(n - k)
        """
        k = int(k)
        if k < 0:
            return 0
        comb = fact(self.n) / (fact(k) * fact(self.n - k))
        q = 1 - self.p
        return comb * (self.p**(k) * q**(self.n - k))

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of “successes”

            k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf


def fact(x):
    """ Factorial function """
    fact = 1
    for i in range(1, x + 1):
        fact *= i
    return fact
