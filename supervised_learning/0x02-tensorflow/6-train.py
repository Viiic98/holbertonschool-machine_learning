#!/usr/bin/env python3
""" Train with tensorflow """
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """ builds, trains, and saves a neural network classifier

        parameters:
        @X_train: is a numpy.ndarray containing the training input data
        @Y_train: is a numpy.ndarray containing the training labels
        @X_valid: is a numpy.ndarray containing the validation input data
        @Y_valid: is a numpy.ndarray containing the validation labels
        @layer_sizes: is a list containing the number of nodes in each
                      layer of the network
        @activations: is a list containing the activation functions for
                      each layer of the network
        @alpha: is the learning rate
        @iterations: is the number of iterations to train over
        @save_path: designates where to save the model
    """
    # Cration of tensors
    x, y = create_placeholders(len(X_train[0]), len(Y_train[0]))
    y_pred = forward_prop(x, layer_sizes, activations)
    # Training
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    # Training
    train_op = create_train_op(loss, alpha)
    saver = tf.train.Saver()
    # Graph collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    # TensorFlow Session
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(iterations + 1):
            ct, at = sess.run([loss, accuracy],
                              feed_dict={x: X_train, y: Y_train})
            cv, av = sess.run([loss, accuracy],
                              feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(ct))
                print("\tTraining Accuracy: {}".format(at))
                print("\tValidation Cost: {}".format(cv))
                print("\tValidation Accuracy: {}".format(av))
            if i == iterations:
                return saver.save(sess, save_path)
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
