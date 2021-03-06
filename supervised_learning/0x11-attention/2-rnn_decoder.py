#!/usr/bin/env python3
""" RNN Decoder """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ RNN Decoder """
    def __init__(self, vocab, embedding, units, batch):
        """ Constructor

            - vocab is an integer representing the size of the output
              vocabulary
            - embedding is an integer representing the dimensionality of
              the embedding vector
            - units is an integer representing the number of hidden
              units in the RNN cell
            - batch is an integer representing the batch size
            public instance attributes:
                embedding - a keras Embedding layer that converts words
                            from the vocabulary into an embedding vector
                gru - a keras GRU layer with units units
                    - Should return both the full sequence of outputs as
                      well as the last hidden state
                    - Recurrent weights should be initialized with
                      glorot_uniform
                F - a Dense layer with vocab units
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """ Call

            - x is a tensor of shape (batch, 1) containing the previous
              word in the target sequence as an index of the target vocabulary
            - s_prev is a tensor of shape (batch, units) containing the
              previous decoder hidden state
            - hidden_states is a tensor of shape (batch, input_seq_len,
              units)containing the outputs of the encoder
            Returns: y, s
                - y is a tensor of shape (batch, vocab) containing the output
                  word as a one hot vector in the target vocabulary
                - s is a tensor of shape (batch, units) containing the new
                  decoder hidden state
        """
        # Self attention
        attention = SelfAttention(self.units)
        context, weights = attention(s_prev, hidden_states)
        # Concat context with x
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], -1)
        # GRU
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)

        return x, state
