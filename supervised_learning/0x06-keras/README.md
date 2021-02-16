# KERAS

![k](https://keras.io/img/logo-k-keras-wb.png)

## Tasks

### [Sequential](./0-sequential.py)
- Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:
    - You are not allowed to use the Input class

### [Input](./1-input.py)
- Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:
    - You are not allowed to use the Sequential class

### [Optimize](./2-optimize.py)
- Write a function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics

### [One Hot](./3-one_hot.py)
- Write a function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix

### [Train](./4-train.py)
- Write a function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descent

### [Validate](./5-train.py)
- Based on [Train](./4-train.py), update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data

### [Early Stopping](./6-train.py)
- Based on [Validate](./5-train.py), update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping

### [Learning Rate Decay](./7-train.py)
Based on [Early Stopping](./6-train.py), update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay

### [Save Only the Best](./8-train.py)
- Based on [Learning Rate Decay](./7-train.py), update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model

### [Save and Load Model](./9-model.py)
- Write the following functions:
    - def save_model(network, filename): saves an entire model:
        - network is the model to save
        - filename is the path of the file that the model should be saved to
        - Returns: None
    - def load_model(filename): loads an entire model:
        - filename is the path of the file that the model should be loaded from
        - Returns: the loaded model

### [Save and Load Weights](./10-weights.py)
- Write the following functions:

    - def save_weights(network, filename, save_format='h5'): saves a model’s weights:
        - network is the model whose weights should be saved
        - filename is the path of the file that the weights should be saved to
        - save_format is the format in which the weights should be saved
        - Returns: None
    - def load_weights(network, filename): loads a model’s weights:
        - network is the model to which the weights should be loaded
        - filename is the path of the file that the weights should be loaded from
        - Returns: None

### [Save and Load Configuration](./11-config.py)
- Write the following functions:

    - def save_config(network, filename): saves a model’s configuration in JSON format:
        - network is the model whose configuration should be saved
        - filename is the path of the file that the configuration should be saved to
        - Returns: None
    - def load_config(filename): loads a model with a specific configuration:
        - filename is the path of the file containing the model’s configuration in JSON format
        - Returns: the loaded model

### [Test](./12-test.py)
- Write a function def test_model(network, data, labels, verbose=True): that tests a neural network

### [Predict](./13-predict.py)
- Write a function def predict(network, data, verbose=False): that makes a prediction using a neural network
