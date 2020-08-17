#!/usr/bin/env python3
""" Evaluation with tensorflow """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a neural network

        @X: is a numpy.ndarray containing the input data to evaluate
        @Y: is a numpy.ndarray containing the one-hot labels for X
        @save_path: is the location to load the model from
        Returns: the network’s prediction, accuracy, and loss, respectively
    """
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        x = tf.get_collection('x')
        y = tf.get_collection('y')
        y_pred = tf.get_collection('y_pred')
        loss = tf.get_collection('loss')
        accuracy = tf.get_collection('accuracy')
        return sess.run([y_pred[0], accuracy[0], loss[0]],
                        feed_dict={x[0]: X, y[0]: Y})
