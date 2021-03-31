# Data Augmentation

## Download TF Datasets
- pip install --user tensorflow-datasets

## Tasks

### [Flip](./0-flip.py)
- Write a function def flip_image(image): that flips an image horizontally.

### [Crop](./1-crop.py)
- Write a function def crop_image(image, size): that performs a random crop of an image.

### [Rotate](./2-rotate.py)
- Write a function def rotate_image(image): that rotates an image by 90 degrees counter-clockwise.

### [Shear](./3-shear.py)
- Write a function def shear_image(image, intensity): that randomly shears an image.

### [Brightness](./4-brightness.py)
- Write a function def change_brightness(image, max_delta): that randomly changes the brightness of an image.

### [Hue](./5-hue.py)
- Write a function def change_hue(image, delta): that changes the hue of an image.

### [PCA Color Augmentation](./100-pca.py)
- Write a function def pca_color(image, alphas): that performs PCA color augmentation as described in the [AlexNet](https://intranet.hbtn.io/rltoken/zEzc_8giX0XkuUTiQsnqXA) paper.
