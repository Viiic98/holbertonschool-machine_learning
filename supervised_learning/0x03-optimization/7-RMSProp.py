#!/usr/bin/env python3
""" RMSProp algorithm """


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using the RMSProp optimization algorithm

        @alpha: is the learning rate
        @beta2: is the RMSProp weight
        @epsilon: is a small number to avoid division by zero
        @var: is a numpy.ndarray containing the variable to be updated
            It's equal to (W or b)
        @grad: is a numpy.ndarray containing the gradient of var
            It's equal to (dw or db)
        @s: is the previous second moment of var
            It's equal to previous (sdw or sdb)
        Returns: the updated variable and the new moment, respectively
    """
    sd = (beta2 * s) + (1 - beta2) * grad**2
    x = var - (alpha * (grad / (sd**(1/2) + epsilon)))
    return x, sd
