#!/usr/bin/env python3
""" PCA Color Augmentation """
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """ performs PCA color augmentation

        - image is a 3D tf.Tensor containing the image to change
        - alphas a tuple of length 3 containing the amount that
          each channel should change
        from https://aparico.github.io/
        Returns the augmented image
    """
    # Load the image(s) as a numpy array
    img = tf.keras.preprocessing.image.img_to_array(image)
    orig_img = tf.keras.preprocessing.image.img_to_array(image)
    # Convert the range of pixel values from 0-255 to 0-1
    img = img / 255.0
    # Flatten the image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # Centering the pixels around their mean
    img_centered = img_rs - np.mean(img_rs, axis=0)
    # Calculate the 3x3 covariance matrix using numpy.cov.
    img_cov = np.cov(img_centered, rowvar=False)
    # Calculate the eigenvalues (3x1 matrix) and
    # eigenvectors (3x3 matrix) of the 3 x3 covariance matrix
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    # sort the eigenvalues and eigenvectors
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]
    # finally get eigenvector matrix [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))
    # Get a 3x1 matrix of eigenvalues multipled by a random
    # variable drawn from a Gaussian distribution with mean=0
    # and sd=0.1 using numpy.random.normal
    m2 = np.zeros((3, 1))
    alpha = np.random.normal(0, 0.1)
    # Create and add the vector (add_vect) that we're going to
    # add to each pixel
    m2[:, 0] = alphas * eig_vals[:]
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):
        orig_img[..., idx] += add_vect[idx]

    # Convert the range of arrays from 0-1 to 0-255
    orig_img = np.clip(orig_img, 0.0, 255.0)
    orig_img = orig_img.astype(np.uint8)

    return orig_img
