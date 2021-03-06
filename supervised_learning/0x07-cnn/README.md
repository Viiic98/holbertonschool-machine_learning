# Convolutional Neural Networks

![c](https://static.packt-cdn.com/products/9781787128422/graphics/B06258_04_01.png)

## Tasks

### [Convolutional Forward Prop](./0-conv_forward.py)
- Function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): that performs forward propagation over a convolutional layer of a neural network

### [Pooling Forward Prop](./1-pool_forward.py)
- Function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs forward propagation over a pooling layer of a neural network

### [Convolutional Back Prop](./2-conv_backward.py)
- Function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): that performs back propagation over a convolutional layer of a neural network

### [Pooling Back Prop](./3-pool_backward.py)
- Function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs back propagation over a pooling layer of a neural network

### [LeNet-5 (Tensorflow)](./4-lenet5.py)
- Function def lenet5(x, y): that builds a modified version of the LeNet-5 architecture using tensorflow

### [LeNet-5 (Keras)](./5-lenet5.py)
- Function def lenet5(X): that builds a modified version of the LeNet-5 architecture using keras