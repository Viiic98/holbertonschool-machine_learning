# Regularization

![r](https://media.geeksforgeeks.org/wp-content/uploads/20190523171704/overfitting_21.png)

## Tasks

### [L2 Regularization Cost](./0-l2_reg_cost.py)
- Write a function def l2_reg_cost(cost, lambtha, weights, L, m): that calculates the cost of a neural network with L2 regularization

### [Gradient Descent with L2 Regularization](./1-l2_reg_gradient_descent.py)
- Write a function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): that updates the weights and biases of a neural network using gradient descent with L2 regularization

### [L2 Regularization Cost](./2-l2_reg_cost.py)
- Write the function def l2_reg_cost(cost): that calculates the cost of a neural network with L2 regularization

### [Create a Layer with L2 Regularization](./3-l2_reg_create_layer.py)
- Write a function def l2_reg_create_layer(prev, n, activation, lambtha): that creates a tensorflow layer that includes L2 regularization

### [Forward Propagation with Dropout](./4-dropout_forward_prop.py)
- Write a function def dropout_forward_prop(X, weights, L, keep_prob): that conducts forward propagation using Dropout

### [Gradient Descent with Dropout](./5-dropout_gradient_descent.py)
- Write a function def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L): that updates the weights of a neural network with Dropout regularization using gradient descent

### [Create a Layer with Dropout](./6-dropout_create_layer.py)
- Write a function def dropout_create_layer(prev, n, activation, keep_prob): that creates a layer of a neural network using dropout

### [Early Stopping](./7-early_stopping.py)
- Write the function def early_stopping(cost, opt_cost, threshold, patience, count): that determines if you should stop gradient descent early
