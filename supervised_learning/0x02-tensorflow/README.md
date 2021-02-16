# Tensorflow

![t](https://camo.githubusercontent.com/c04e16c05de80dadbdc990884672fc941fdcbbfbb02b31dd48c248d010861426/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f736f6369616c2e706e67)

## Tasks

### [Placeholders](./0-create_placeholders.py)
- Write the function def create_placeholders(nx, classes): that returns two placeholders, x and y, for the neural network

### [Layers](./1-create_layer.py)
- Write the function def create_layer(prev, n, activation)

### [Forward Propagation](./2-forward_prop.py)
- Write the function def forward_prop(x, layer_sizes=[], activations=[]): that creates the forward propagation graph for the neural network

### [Accuracy](./3-calculate_accuracy.py)
- Write the function def calculate_accuracy(y, y_pred): that calculates the accuracy of a prediction

### [Loss](./4-calculate_loss.py)
- Write the function def calculate_loss(y, y_pred): that calculates the softmax cross-entropy loss of a prediction

### [Train_Op](./5-create_train_op.py)
- Write the function def create_train_op(loss, alpha): that creates the training operation for the network

### [Train](./6-train.py)
- Write the function def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"): that builds, trains, and saves a neural network classifier

### [Evaluate](./7-evaluate.py)
- Write the function def evaluate(X, Y, save_path): that evaluates the output of a neural network
