#!/usr/bin/env python3
""" class Yolo that uses the Yolo v3 algorithm to perform object detection """
import tensorflow.keras as K
import numpy as np
import cv2
import glob


def sigmoid(x):
    """ Sigmoid function """
    return 1/(1 + np.exp(-x))


class Yolo:
    """ YOLO Class """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Constructor

        - model_path is the path to where a Darknet Keras model is stored
        - classes_path is the path to where the list of class names used
          for the Darknet model, listed in order of index, can be found
        - class_t is a float representing the box score threshold for the
          initial filtering step
        - nms_t is a float representing the IOU threshold for non-max
          suppression
        - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
          containing all of the anchor boxes:
        - outputs is the number of outputs (predictions) made by the
          Darknet model
        - anchor_boxes is the number of anchor boxes used for each
          prediction
        - 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """

        - outputs is a list of numpy.ndarrays containing the predictions
          from the Darknet model for a single image:
        - Each output will have the shape (grid_height, grid_width,
         anchor_boxes, 4 + 1 + classes)
        - grid_height & grid_width => the height and width of
          the grid used for the output
        - anchor_boxes => the number of anchor boxes used
        - 4 => (t_x, t_y, t_w, t_h)
        - 1 => box_confidence
        - classes => class probabilities for all classes
        - image_size is a numpy.ndarray containing the image’s original
          size [image_height, image_width]
        - Returns a tuple of (boxes, box_confidences, box_class_probs):
        - boxes: a list of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, 4) containing the processed boundary
          boxes for each output, respectively:
        - 4 => (x1, y1, x2, y2)
        - (x1, y1, x2, y2) should represent the boundary box relative
          to original image
        - box_confidences: a list of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, 1) containing the box confidences for
          each output, respectively
        - box_class_probs: a list of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, classes) containing the box’s class
          probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        height, width = image_size
        i = 0
        for output in outputs:
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchors = self.anchors[i]
            box = output[:, :, :, :4]
            for cx in range(len(output)):
                for cy in range(len(output[cx])):
                    for anchor in range(len(output[cx][cy])):
                        pw, ph = anchors[anchor]
                        tx = output[cx, cy, anchor, 0]
                        ty = output[cx, cy, anchor, 1]
                        tw = output[cx, cy, anchor, 2]
                        th = output[cx, cy, anchor, 3]
                        bx = sigmoid(tx) + cx
                        bx = bx / grid_width
                        by = sigmoid(ty) + cy
                        by = by / grid_height
                        bw = pw * np.exp(tw)
                        bw = bw / self.model.input.shape[1].value
                        bh = ph * np.exp(th)
                        bh = bh / self.model.input.shape[2].value
                        x1 = (bx - bw / 2) * width
                        y1 = (by - bh / 2) * height
                        x2 = (bx + bw / 2) * width
                        y2 = (by + bh / 2) * height
                        box[cx, cy, anchor, :] = x1, y1, x2, y2
            i += 1
            boxes.append(box)
            box_confidences.append(sigmoid(output[:, :, :, 4:5]))
            box_class_probs.append(sigmoid(output[:, :, :, 5:]))
        return boxes, box_confidences, box_class_probs

    @staticmethod
    def load_images(folder_path):
        """ Load images

            - folder_path: a string representing the path to the
              folder holding all the images to load
            Returns a tuple of (images, image_paths):
                - images: a list of images as numpy.ndarrays
                - image_paths: a list of paths to the individual
                  images in images
        """
        images = []
        image_paths = []
        for img in glob.glob(folder_path + '/*'):
            image_paths.append(img)
            img = cv2.imread(img, 1)
            images.append(img)
        return images, image_paths

    def preprocess_images(self, images):
        """ Preprocess images

            - images: a list of images as numpy.ndarrays
            - Resize the images with inter-cubic interpolation
            - Rescale all images to have pixel values in the range [0, 1]
            Returns a tuple of (pimages, image_shapes):
                - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                  containing all of the preprocessed images
                    - ni: the number of images that were preprocessed
                    - input_h: the input height for the Darknet model Note:
                      this can vary by model
                    - input_w: the input width for the Darknet model Note:
                      this can vary by model
                    - 3: number of color channels
                - image_shapes: a numpy.ndarray of shape (ni, 2) containing
                  the original height and width of the images
                    - 2 => (image_height, image_width)
        """
        image_shapes = np.zeros((len(images), 2))
        width = self.model.input_shape[1]
        height = self.model.input_shape[2]
        dim = (width, height)
        pimages = []
        i = 0
        for img in images:
            org_width = img.shape[0]
            org_height = img.shape[1]
            image_shapes[i] = org_width, org_height
            resize = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
            pimages.append(resize / 255)
            i += 1
        return np.array(pimages), np.array(image_shapes)
