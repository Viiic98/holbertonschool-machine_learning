#!/usr/bin/env python3
""" Normal distribution """


class Normal():
    """ Normal distribution """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Represents a normal distribution

            Data is a list of the data to be used to estimate the distribution
            Mean is the mean of the distribution
            Stddev is the standard deviation of the distribution
        """
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                n = 0
                for i in data:
                    n += (i - self.mean)**2
                self.stddev = (n / len(data))**(1/2)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if type(data) is list and len(data) < 2:
                raise ValueError("data must contain multiple values")
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value

            z = (x - mean) / standard deviation
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score

            x = (z * standard deviation) + mean
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ Calculates the value of the PDF for a given x-value

            PDF = (1 / (standard deviation * (2 * pi)**(1/2)))
            PDF = PDF * (exponential**((-1/2)
            PDF = PDF * ((x - mean) / standard deviation)**2)
        """
        pdf = (1 / (self.stddev * (2 * self.pi)**(1/2)))
        pdf = pdf * (self.e**((-1/2) * (self.z_score(x)**2)))
        return pdf

    def cdf(self, x):
        """ Calculates the value of the CDF for a given x-value

            CDF =
        """
        cdf = 0
        for i in range(x):
            cdf += (self.e**((-i**2)/2))
        return cdf
