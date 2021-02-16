# Error Analysis

![E](https://imgs.xkcd.com/comics/machine_learning.png)

## Tasks

### [Create Confusion](./0-create_confusion.py)
- Write the function def create_confusion_matrix(labels, logits): that creates a confusion matrix

### [Sensitivity](./1-sensitivity.py)
- Write the function def sensitivity(confusion): that calculates the sensitivity for each class in a confusion matrix

### [Precision](./2-precision.py)
- Write the function def precision(confusion): that calculates the precision for each class in a confusion matrix

### [Specificity](./3-specificity.py)
- Write the function def specificity(confusion): that calculates the specificity for each class in a confusion matrix

### [F1 score](./4-f1_score.py)
- Write the function def f1_score(confusion): that calculates the F1 score of a confusion matrix

### [Dealing with Error](./5-error_handling)
- In the text file 5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C)

Scenarios:

1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance

Approaches:

A) Train more

B) Try a different architecture

C. Get more data

D. Build a deeper network

E. Use regularization

F. Nothing

### [Compare and Contrast](./6-compare_and_contrast)
- Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file 6-compare_and_contrast
