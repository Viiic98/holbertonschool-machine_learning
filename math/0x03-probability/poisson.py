#!/usr/bin/env python3
""" Poisson Distribution """


class Poisson:
    """ Poisson distribution """
    def __init__(self, data=None, lambtha=1.):
        """ Class Constructor

            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        if lambtha < 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = float(lambtha)
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of “successes”

            k: number of “successes”
            pmf = ((lambtha ** k) * (e ** -lambtha)) / k!
        """
        if type(k) is not int:
            k = int(k)

        # Factorial
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        pmf = (self.lambtha ** (k)) * (2.7182818285 ** (-self.lambtha))
        pmf = pmf / fact
        return pmf
