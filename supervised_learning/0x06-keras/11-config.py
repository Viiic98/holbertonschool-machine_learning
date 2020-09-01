#!/usr/bin/env python3
""""""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model’s configuration in JSON format

        @network: is the model whose configuration should be saved
        @filename: is the path of the file that the configuration
                   should be saved to
        Returns: None
    """
    json_conf = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(json_conf)


def load_config(filename):
    """ loads a model with a specific configuration

        @filename: is the path of the file containing the model’s
                   configuration in JSON format
        Returns: the loaded model
    """
    with open(filename, "r") as json_file:
        json_conf = json_file.read()
    network = K.models.model_from_json(json_conf)
    return network
