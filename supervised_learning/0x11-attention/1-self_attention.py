#!/usr/bin/env python3
""" Self Attention """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Self Attention """
    def __init__(self, units):
        """ Class constructor

            - units is an integer representing the number of hidden units
              in the alignment model
            public instance attributes:
                W - a Dense layer with units units, to be applied to the
                    previous decoder hidden state
                U - a Dense layer with units units, to be applied to the
                    encoder hidden states
                V - a Dense layer with 1 units, to be applied to the tanh
                    of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ Call

            - s_prev is a tensor of shape (batch, units) containing the
              previous decoder hidden state
            - hidden_states is a tensor of shape (batch, input_seq_len,
              units)containing the outputs of the encoder
            Returns: context, weights
                - context is a tensor of shape (batch, units) that
                  contains the context vector for the decoder
                - weights is a tensor of shape (batch, input_seq_len, 1)
                  that contains the attention weights
        """
        prev = self.W(tf.expand_dims(s_prev, 1))
        enc = self.U(hidden_states)
        e = self.V(tf.tanh(prev + enc))
        w = tf.nn.softmax(e, 1)
        context = e * hidden_states
        return tf.math.reduce_sum(context, 1), w
