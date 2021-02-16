# Optimization

![o](https://blog.paperspace.com/content/images/2018/06/optimizers7.gif)

## Tasks

### [Normalization Constants](./0-norm_constants.py)
- Write the function def normalization_constants(X): that calculates the normalization (standardization) constants of a matrix

### [Normalize](./1-normalize.py)
- Write the function def normalize(X, m, s): that normalizes (standardizes) a matrix

### [Shuffle Data](./2-shuffle_data.py)
- Write the function def shuffle_data(X, Y): that shuffles the data points in two matrices the same way

### [Mini-Batch](./3-mini_batch.py)
- Write the function def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"): that trains a loaded neural network model using mini-batch gradient descent

### [Moving Average](./4-moving_average.py)
- Write the function def moving_average(data, beta): that calculates the weighted moving average of a data set

### [Momentum](./5-momentum.py)
- Write the function def update_variables_momentum(alpha, beta1, var, grad, v): that updates a variable using the gradient descent with momentum optimization algorithm

### [Momentum Upgraded](./6-momentum.py)
- Write the function def create_momentum_op(loss, alpha, beta1): that creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm

### [RMSProp](./7-RMSProp.py)
- Write the function def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s): that updates a variable using the RMSProp optimization algorithm

### [RMSProp Upgraded](./8-RMSProp.py)
- Write the function def create_RMSProp_op(loss, alpha, beta2, epsilon): that creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm

### [Adam](./9-Adam.py)
- Write the function def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t): that updates a variable in place using the Adam optimization algorithm

### [Adam Upgraded](./10-Adam.py)
- Write the function def create_Adam_op(loss, alpha, beta1, beta2, epsilon): that creates the training operation for a neural network in tensorflow using the Adam optimization algorithm

### [Learning Rate Decay](./11-learning_rate_decay.py)
- Write the function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that updates the learning rate using inverse time decay in numpy

### [Learning Rate Decay Upgraded](./12-learning_rate_decay.py)
- Write the function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that creates a learning rate decay operation in tensorflow using inverse time decay

### [Batch Normalization](./13-batch_norm.py)
- Write the function def batch_norm(Z, gamma, beta, epsilon): that normalizes an unactivated output of a neural network using batch normalization

### [Batch Normalization Upgraded](./14-batch_norm.py)
- Write the function def create_batch_norm_layer(prev, n, activation): that creates a batch normalization layer for a neural network in tensorflow

### [Put it all together and what do you get?](./15-model.py)
- Write the function def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'): that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization 
