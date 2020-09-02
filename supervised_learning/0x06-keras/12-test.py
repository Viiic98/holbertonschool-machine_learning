#!/usr/bin/env python3
""" Test a neural network """


def test_model(network, data, labels, verbose=True):
    """ tests a neural network

        @network: is the network model to test
        @data: is the input data to test the model with
        @labels: are the correct one-hot labels of data
        @verbose: is a boolean that determines if output
                  should be printed during the testing process
        Returns: the loss and accuracy of the model with the
                 testing data, respectively
    """
    results = network.evaluate(data, labels, verbose=verbose)
    return results
