# Object Detection

![obj](https://miro.medium.com/max/739/1*IrptRDRG8IL9o-55BKjbLA.png)

## Download and Use OpenCV 4.1.x
- pip install --user opencv-python
- To use it import cv2

## Tasks

### [Initialize Yolo](./0-yolo.py)
- Write a class Yolo that uses the Yolo v3 algorithm to perform object detection

### [Process Outputs](./1-yolo.py)
- Write a class Yolo (Based on [Initialize Yolo](./0-yolo.py)):
    - Add the public method def process_outputs(self, outputs, image_size)

### [Filter Boxes](./2-yolo.py)
- Write a class Yolo (Based on [Process Outputs](./1-yolo.py)):
    - Add the public method def filter_boxes(self, boxes, box_confidences, box_class_probs)

### [Non-max Suppression](./3-yolo.py)
- Write a class Yolo (Based on [Filter Boxes](./2-yolo.py)):
    - Add the public method def non_max_suppression(self, filtered_boxes, box_classes, box_scores)

### [Load images](./4-yolo.py)
- Write a class Yolo (Based on [Non-max Suppression](./3-yolo.py)):
    - Add the static method def load_images(folder_path):

### [Preprocess images](./5-yolo.py)
- Write a class Yolo (Based on [Load images](./4-yolo.py)):
    - Add the public method def preprocess_images(self, images)

### [Show boxes](./6-yolo.py)
- Write a class Yolo (Based on [Preprocess images](./5-yolo.py)):
    - Add the public method def show_boxes(self, image, boxes, box_classes, box_scores, file_name)

### [Predict](./7-yolo.py)
- Write a class Yolo (Based on [Show boxes](./6-yolo.py)):
    - Add the public method def predict(self, folder_path)
