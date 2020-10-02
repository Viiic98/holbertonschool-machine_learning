#!/usr/bin/env python3
""" Train transef learning model """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ pre-processes the data for your model

        X: is a numpy.ndarray of shape (m, 32, 32, 3) containing
           the CIFAR 10 data, where m is the number of data points
        Y: is a numpy.ndarray of shape (m,) containing the CIFAR
           10 labels for X
        Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # normalize to range 0-1
    X = K.utils.normalize(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


def transfer_model(X_t, Y_t, X_v, Y_v):
    """ Create and train a transfer learning model

        Based on DenseNet121
    """
    # input tensor
    inputs = K.Input((32, 32, 3))
    # Resize images to 224x224
    x = K.layers.Lambda(lambda x: K.backend.resize_images(x, 7, 7,
                        "channels_last"))(inputs)
    # Get model DenseNet 121
    base_model = K.applications.DenseNet121(weights='imagenet',
                                            include_top=False,
                                            input_shape=(224, 224, 3))

    # Set juss the last half of layers as trainable
    x = base_model(x)
    # Flat to prepare to FC
    x = K.layers.Flatten()(x)
    # Batch norm
    x = K.layers.BatchNormalization()(x)
    # Fully connected layer
    x = K.layers.Dense(128, activation='relu')(x)
    # Dropout
    x = K.layers.Dropout(0.5)(x)
    # Batchnorm
    x = K.layers.BatchNormalization()(x)
    # Fully connected layer
    x = K.layers.Dense(64, activation='relu')(x)
    # Dropout
    x = K.layers.Dropout(0.5)(x)
    # Batchnorm
    x = K.layers.BatchNormalization()(x)
    # Softmax function to classify cifar 10
    outputs = K.layers.Dense(units=10, activation='softmax')(x)
    # Create the model
    model = K.models.Model(inputs, outputs)
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=K.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    # Train model
    model.fit(X_t, Y_t, batch_size=32, epochs=5, verbose=1,
              validation_data=(X_v, Y_v))
    # Save model
    model.save('cifar10.h5')


K.learning_phase = K.backend.learning_phase

(X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
X_train, Y_train = preprocess_data(X_train, Y_train)
X_valid, Y_valid = preprocess_data(X_valid, Y_valid)
transfer_model(X_train, Y_train, X_valid, Y_valid)
