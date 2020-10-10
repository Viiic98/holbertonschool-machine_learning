#!/usr/bin/env python3
""" class Yolo that uses the Yolo v3 algorithm to perform object detection """
import os
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
                        bx = sigmoid(tx) + cy
                        bx = bx / grid_width
                        by = sigmoid(ty) + cx
                        by = by / grid_height
                        bw = pw * np.exp(tw)
                        bw = bw / self.model.input.shape[1].value
                        bh = ph * np.exp(th)
                        bh = bh / self.model.input.shape[2].value
                        x1 = (bx - bw / 2)
                        x2 = (x1 + bw)
                        y1 = (by - bh / 2)
                        y2 = (y1 + bh)
                        box[cx, cy, anchor, 0] = x1 * width
                        box[cx, cy, anchor, 1] = y1 * height
                        box[cx, cy, anchor, 2] = x2 * width
                        box[cx, cy, anchor, 3] = y2 * height
            i += 1
            boxes.append(box)
            box_confidences.append(sigmoid(output[:, :, :, 4:5]))
            box_class_probs.append(sigmoid(output[:, :, :, 5:]))
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Filter box

            - boxes: a list of numpy.ndarrays of shape (grid_height,
                     grid_width, anchor_boxes, 4) containing the
                     processed boundary boxes for each output, respectively
            - box_confidences: a list of numpy.ndarrays of shape (grid_height,
                               grid_width, anchor_boxes, 1) containing the
                               processed box confidences for each output,
                               respectively
            - box_class_probs: a list of numpy.ndarrays of shape (grid_height,
                               grid_width, anchor_boxes, classes) containing
                               the processed box class probabilities for
                               each output, respectively
            Returns a tuple of (filtered_boxes, box_classes, box_scores):
                - filtered_boxes: a numpy.ndarray of shape (?, 4) containing
                                  all of the filtered bounding boxes:
                - box_classes: a numpy.ndarray of shape (?,) containing the
                               class number that each box in filtered_boxes
                               predicts, respectively
                - box_scores: a numpy.ndarray of shape (?) containing the
                              box scores for each box in filtered_boxes,
                              respectively
        """
        filtered_scores = []
        filtered_boxes = []
        filtered_classes = []
        for i in range(len(boxes)):
            box_scores = box_confidences[i] * box_class_probs[i]
            box_classes = np.argmax(box_scores, axis=-1)
            box_scores = np.max(box_scores, axis=-1)
            filtering_mask = box_scores >= self.class_t

            filtered_scores += box_scores[filtering_mask].tolist()
            filtered_boxes += boxes[i][filtering_mask].tolist()
            filtered_classes += box_classes[filtering_mask].tolist()
        filtered_boxes = np.array(filtered_boxes)
        filtered_classes = np.array(filtered_classes)
        filtered_scores = np.array(filtered_scores)
        return filtered_boxes, filtered_classes, filtered_scores

    def iou(self, box1, box2):
        """ Intersection over union """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
        if w_intersection <= 0 or h_intersection <= 0:  # No overlap
            return 0
        i = w_intersection * h_intersection
        u = w1 * h1 + w2 * h2 - i  # Union = Total Area - I
        return i / u

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Non-max Suppression

            - filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
                              of the filtered bounding boxes:
            - box_classes: a numpy.ndarray of shape (?,) containing the class
                           number for the class that filtered_boxes predicts,
                           respectively
            - box_scores: a numpy.ndarray of shape (?) containing the box
                          scores for each box in filtered_boxes, respectively
            Returns a tuple of (box_predictions, predicted_box_classes,
                                predicted_box_scores):
                - box_predictions: a numpy.ndarray of shape (?, 4) containing
                                   all of the predicted bounding boxes ordered
                                   by class and box score
                - predicted_box_classes: a numpy.ndarray of shape (?,)
                                         containing the class number for
                                         box_predictions ordered by class
                                         and box score, respectively
                - predicted_box_scores: a numpy.ndarray of shape (?)
                                        containing the box scores for
                                        box_predictions ordered by class and
                                        box score, respectively
        """
        pick = []
        idxs = np.lexsort((box_scores, -box_classes))
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
            for pos in range(last):
                j = idxs[pos]
                if box_classes[i] == box_classes[j]:
                    if self.iou(filtered_boxes[i], filtered_boxes[j]) > \
                            self.nms_t:
                        suppress.append(pos)
            idxs = np.delete(idxs, suppress)
        return filtered_boxes[pick], box_classes[pick], box_scores[pick]

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
        image_shapes = []
        width = self.model.input_shape[1]
        height = self.model.input_shape[2]
        dim = (width, height)
        pimages = []
        i = 0
        for img in images:
            org_width = img.shape[0]
            org_height = img.shape[1]
            image_shapes.append((org_width, org_height))
            resize = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
            pimages.append(resize / 255)
            i += 1
        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """

        - image: a numpy.ndarray containing an unprocessed image
        - boxes: a numpy.ndarray containing the boundary boxes for the image
        - box_classes: a numpy.ndarray containing the class indices for each
          box
        - box_scores: a numpy.ndarray containing the box scores for each box
        - file_name: the file path where the original image is stored
        - Displays the image with all boundary boxes, class names, and
          box scores (see example below)
            - Boxes should be drawn as with a blue line of thickness 2
            - Class names and box scores should be drawn above each box in red
                - Box scores should be rounded to 2 decimal places
                - Text should be written 5 pixels above the top left corner
                  of the box
                - Text should be written in FONT_HERSHEY_SIMPLEX
                - Font scale should be 0.5
                - Line thickness should be 1
                - You should use LINE_AA as the line type
            - The window name should be the same as file_name
            - If  the s key is pressed:
                - The image should be saved in the directory detections,
                  located in the current directory
                - If detections does not exist, create it
                - The saved image should have the file name file_name
                - The image window should be closed
            - If any key besides s is pressed, the image window should be
              closed without saving
        """
        rec_color = (255, 0, 0)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            x1y1 = (x1, y1)
            x2y2 = (x2, y2)
            cv2.rectangle(image, x1y1, x2y2, color=rec_color, thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            x = int(boxes[i][0])
            y = int(boxes[i][1]) - 5
            name_class = self.class_names[box_classes[i]]
            score = str(round(box_scores[i], 2))
            line = cv2.LINE_AA
            cv2.putText(image, name_class + ' ' + score, (x, y),
                        font, 0.5, (0, 0, 255), thickness=1, lineType=line)
        cv2.imshow(file_name, image)
        k = cv2.waitKey(0)
        if k == ord('s'):
            if not os.path.exists('detections/'):
                os.mkdir('detections')
            cv2.imwrite('detections/' + file_name, image)
        cv2.destroyAllWindows()
