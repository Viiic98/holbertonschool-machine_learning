# Convolutions and Pooling

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/ed9ca14839ad0201f19e.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210213T021455Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8641c99f3d1989c2c942489fa1eceb29f804d2cb40455adb0363fd7c5162b751)

## Tasks

### [Valid Convolution](./0-convolve_grayscale_valid.py)
- Write a function def convolve_grayscale_valid(images, kernel): that performs a valid convolution on grayscale images

### [Same Convolution](./1-convolve_grayscale_same.py)
- Write a function def convolve_grayscale_same(images, kernel): that performs a same convolution on grayscale images

### [Convolution with Padding](./2-convolve_grayscale_padding.py)
- Write a function def convolve_grayscale_padding(images, kernel, padding): that performs a convolution on grayscale images with custom padding

### [Strided Convolution](./3-convolve_grayscale.py)
- Write a function def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on grayscale images

### [Convolution with Channels](./4-convolve_channels.py)
- Write a function def convolve_channels(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on images with channels

### [Multiple Kernels](./5-convolve.py)
- Write a function def convolve(images, kernels, padding='same', stride=(1, 1)): that performs a convolution on images using multiple kernels

### [Pooling](./6-pool.py)
- Write a function def pool(images, kernel_shape, stride, mode='max'): that performs pooling on images
