#!/usr/bin/env python3
""" Neural network using Model class """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library

        @nx: is the number of input features to the network
        @layers: is a list containing the number of nodes in each
                 layer of the network
        @activations: is a list containing the activation functions used
                      for each layer of the network
        @lambtha: is the L2 regularization parameter
        @keep_prob: is the probability that a node will be kept for
                    dropout
        You are not allowed to use the Sequential class
        Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    inp = inputs
    for i in range(len(layers)):
        if i + 1 < len(layers):
            layer = K.layers.Dense(layers[i],
                                   activation=activations[i],
                                   kernel_regularizer=K.regularizers.l2(
                                                        lambtha))(inp)
            dropout = (K.layers.Dropout(1 - keep_prob))(layer)
            inp = dropout
        else:
            layer = K.layers.Dense(layers[i],
                                   activation=activations[i],
                                   kernel_regularizer=K.regularizers.l2(
                                                        lambtha))(inp)
            output = layer
    model = K.Model(inputs=inputs, outputs=output)
    return model
