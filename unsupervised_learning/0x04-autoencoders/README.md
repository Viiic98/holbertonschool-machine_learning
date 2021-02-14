# Autoencoders

![A](https://miro.medium.com/max/3524/1*oUbsOnYKX5DEpMOK3pH_lg.png)

## Tasks

### ["Vanilla" Autoencoder](./0-vanilla.py)
- Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates an autoencoder
    - The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    - All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid

### [Sparse Autoencoder](./1-sparse.py)
- Write a function def autoencoder(input_dims, hidden_layers, latent_dims, lambtha): that creates a sparse autoencoder
    - The sparse autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    - All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid

### [Convolutional Autoencoder](./2-convolutional.py)
- Write a function def autoencoder(input_dims, filters, latent_dims): that creates a convolutional autoencoder
    - The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    
### [Variational Autoencoder](./3-variational.py)
- Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates a variational autoencoder
    - The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    - All layers should use a relu activation except for the mean and log variance layers in the encoder, which should use None, and the last layer in the decoder, which should use sigmoid