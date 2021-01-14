#!/usr/bin/env python3
""" Masks """
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """ creates all masks for training/validation """
    inputs = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    target = tf.cast(tf.math.equal(target, 0), tf.float32)
    # Encoder padding mask
    enc_padding_mask = inputs[:, tf.newaxis, tf.newaxis, :]

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = inputs[:, tf.newaxis, tf.newaxis, :]

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    dec_target_mask = target[:, tf.newaxis, tf.newaxis, :]
    x, y = target.shape
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((x, 1, y, y)), -1, 0)
    combined_mask = tf.maximum(dec_target_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask
