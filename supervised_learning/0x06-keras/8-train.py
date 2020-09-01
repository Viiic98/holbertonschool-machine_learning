#!/usr/bin/env python3
""" Training keras model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """  trains a model using mini-batch gradient descent

        @network: is the model to train
        @data: is a numpy.ndarray of shape (m, nx) containing the
               input data
        @labels: is a one-hot numpy.ndarray of shape (m, classes)
                 containing the labels of data
        @batch_size: is the size of the batch used for mini-batch
                     gradient descent
        @epochs: is the number of passes through data for mini-batch
                 gradient descent
        @verbose: is a boolean that determines if output should be
                  printed during training
        @shuffle: is a boolean that determines whether to shuffle the
                  batches every epoch. Normally, it is a good idea to
                  shuffle, but for reproducibility, we have chosen to
                  set the default to False.
        @validation_data: is the data to validate the model with, if not None
        @early_stopping: is a boolean that indicates whether early stopping
                         should be used
            - early stopping should only be performed if validation_data exists
            - early stopping should be based on validation loss
        @patience: is the patience used for early stopping
        @learning_rate_decay: is a boolean that indicates whether learning rate
                              decay should be used
            - learning rate decay should only be performed if validation_data
              exists
            - the decay should be performed using inverse time decay
            - the learning rate should decay in a stepwise fashion after each
              epoch
            - each time the learning rate updates, Keras should print a message
        @alpha: is the initial learning rate
        @decay_rate: is the decay rate
        @save_best: is a boolean indicating whether to save the model after
                    each epoch if it is the best
            - a model is considered the best if its validation loss is the
              lowest that the model has obtained
        @filepath: is the file path where the model should be saved
        Returns: the History object generated after training the model
    """
    if filepath:
        call_backs = [K.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                  save_best_only=True)]
    else:
        call_backs = []
    if validation_data:
        if early_stopping or learning_rate_decay:
            if early_stopping:
                call_backs.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=patience))
            if learning_rate_decay:
                def scheduler(epoch):
                    """ Scheduler function """
                    return alpha * 1/(1 + decay_rate * epoch)
                call_backs.append(
                            K.callbacks.LearningRateScheduler(scheduler,
                                                              verbose=True))
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=call_backs)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              call_backs=call_backs)
    return history
