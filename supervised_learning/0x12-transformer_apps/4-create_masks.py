#!/usr/bin/env python3
""" Masks """
import tensorflow as tf


def create_masks(inputs, target):
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]
    size = target.shape[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones(
        (target.shape[0], 1, size, size)), -1, 0)
    return encoder_mask, look_ahead_mask, decoder_mask
