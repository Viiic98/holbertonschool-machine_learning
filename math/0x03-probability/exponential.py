#!/usr/bin/env python3
""" Class that represents an exponential distribution """


class Exponential():
    """ Exponential class """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Represents an exponential distribution

            Data is a list of the data to be used to estimate the distribution
            Lambtha is the expected number of occurences in a given time frame
        """
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (1 / sum(data)) * 100
        else:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period

            PDF = lambtha * exponential ** (-lambtha * x)
        """
        return self.lambtha * self.e**(-self.lambtha * x)

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period

            CDF = 1 - exponential ** (-lambtha * x)
        """
        return 1 - self.e**(-self.lambtha * x)
