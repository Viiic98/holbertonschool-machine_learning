#!/usr/bin/env python3
""""""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the gradient descent
        with momentum optimization algorithm

        @alpha: is the learning rate
        @beta1: is the momentum weight
        @var: is a numpy.ndarray containing the variable to be updated
            It's equal to W or b
        @grad: is a numpy.ndarray containing the gradient of var
            It's equal to dw or db
        @v: is the previous first moment of var
        Returns: the updated variable and the new moment, respectively

        Formula:
            vdw or vdb = beta * (vdw or vdb) + 1 - beta * (dw or db)
    """
    vd = (beta1 * v) + ((1 - beta1) * grad)
    x = var - (alpha * v)
    return x, vd
