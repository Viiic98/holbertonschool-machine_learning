# Deep Convolutional Architectures

### Tasks

#### [Inception Block](./0-inception_block.py)
- Write a function def inception_block(A_prev, filters): that builds an inception block as described in [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf)

#### [Inception Network](./1-inception_network.py)
- Write a function def inception_network(): that builds the inception network as described in [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf)

#### [Identity Block](./2-identity_block.py)
- Write a function def identity_block(A_prev, filters): that builds an identity block as described in [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf)

#### [Projection Block](./3-projection_block.py)
- Write a function def projection_block(A_prev, filters, s=2): that builds a projection block as described in [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf)

#### [ResNet-50](./4-resnet50.py)
- Write a function def resnet50(): that builds the ResNet-50 architecture as described in [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf)
#### [Dense Block](./5-dense_block.py)
- Write a function def dense_block(X, nb_filters, growth_rate, layers): that builds a dense block as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

#### [Transition Layer](./6-transition_layer.py)
- Write a function def transition_layer(X, nb_filters, compression): that builds a transition layer as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

#### [DenseNet-121](./7-densenet121.py)
- Write a function def densenet121(growth_rate=32, compression=1.0): that builds the DenseNet-121 architecture as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
