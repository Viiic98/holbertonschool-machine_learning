#!/usr/bin/env python3
""" Adam Algorithm """


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates a variable in place using the Adam optimization algorithm

        @alpha: is the learning rate
        @beta1: is the weight used for the first moment
        @beta2: is the weight used for the second moment
        @epsilon: is a small number to avoid division by zero
        @var: is a numpy.ndarray containing the variable to be updated
            It's equal to (W or b)
        @grad: is a numpy.ndarray containing the gradient of var
            It's equal to (dw or db)
        @v: is the previous first moment of var
            It's equal to (Vdw1 or Vdb1)
        @s: is the previous second moment of var
            It's equal to (Vdw2 or Vdb2)
        @t: is the time step used for bias correction
        Returns: the updated variable, the new first moment, and the
                 new second moment, respectively
    """
    vd = beta1 * v + (1 - beta1) * grad
    sd = beta2 * s + (1 - beta2) * grad**2
    vdc = vd / (1 - beta1**t)
    sdc = sd / (1 - beta2**t)
    x = var - alpha * (vdc / (sdc**(1/2) + epsilon))
    return x, vd, sd
