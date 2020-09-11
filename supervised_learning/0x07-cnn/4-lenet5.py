#!/usr/bin/env python3
""" LeNet_5 using tensorflow """
import tensorflow as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5
        architecture using tensorflow

        @x: is a tf.placeholder of shape (m, 28, 28, 1)
            containing the input images for the network
            - m is the number of images
        @y: is a tf.placeholder of shape (m, 10) containing
            the one-hot labels for the network
        - The model should consist of the following layers in order:
            - Convolutional layer with 6 kernels of shape 5x5 with
              same padding
            - Max pooling layer with kernels of shape 2x2 with 2x2
              strides
            - Convolutional layer with 16 kernels of shape 5x5 with
              valid padding
            - Max pooling layer with kernels of shape 2x2 with 2x2
              strides
            - Fully connected layer with 120 nodes
            - Fully connected layer with 84 nodes
            - Fully connected softmax output layer with 10 nodes
        - All layers requiring initialization should initialize their
          kernels with the he_normal initialization method:
                - tf.contrib.layers.variance_scaling_initializer()
        - All hidden layers requiring activation should use the relu
          activation function
        - you may import tensorflow as tf
        - you may NOT use tf.keras
        Returns:
            - a tensor for the softmax activated output
            - a training operation that utilizes Adam optimization
              (with default hyperparameters)
            - a tensor for the loss of the netowrk
            - a tensor for the accuracy of the network
    """
    # Kernel initializer
    kernel_init = tf.contrib.layers.variance_scaling_initializer()

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
            inputs=x,
            filters=6,
            kernel_size=5,
            kernel_initializer=kernel_init,
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=16,
          kernel_size=5,
          kernel_initializer=kernel_init,
          padding="valid",
          activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)

    # Reshaping output into a single dimention array for input
    # to fully connected layer
    pool2_flat = tf.layers.Flatten()(pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.dense(inputs=pool2_flat, units=120,
                             kernel_initializer=kernel_init,
                             activation=tf.nn.relu)

    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.dense(inputs=dense1, units=84,
                             kernel_initializer=kernel_init,
                             activation=tf.nn.relu)

    # Output layer, 10 neurons for each digit
    logits = tf.layers.dense(inputs=dense2, units=10,
                             kernel_initializer=kernel_init)

    # Softmax function
    softmax = tf.nn.softmax(logits)

    # Compute the cross-entropy loss function
    # loss = tf.losses.softmax_cross_entropy(y, softmax)
    cost = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                      labels=y))

    # Training operation with Adam Optimization
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cost)

    # For testing and prediction
    predictions = tf.argmax(softmax, axis=1)
    correct_prediction = tf.equal(tf.argmax(y, 1), predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, cost, accuracy
