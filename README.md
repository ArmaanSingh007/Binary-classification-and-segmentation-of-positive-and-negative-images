Disclaimer: https://github.com/ArmaanSingh007/Binary-classification-and-segmentation-of-positive-and-negative-images © 2024 by Armaan Singh, Kintur raja is licensed under Creative Commons Attribution 4.0 International .

For dataset: Please download it from this link ' https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification '.

Important note: When we run data in python it take too much time (Without using any hardware like GPU or external GPUs) to run and didn't show the output while doing task-1 because of huge data(images), so we need to trained our data through GPU (CUDA-GTX 1660 TI WITH MAX-Q DESIGN).we classified image by two method Resnet and Alexnet in which Resnet turns to be the great choice among them but have high computational cost.

For dataset: Please download it from this link ' https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification '( In case you have some problem for dataset).

Goal: The main goal of this task is to develop a comprehensive pipeline that starts with binary classification of images to identify those containing cracks, and then moves on to segment that identified cracks. The pipeline should leverage machine learning for classification and unsupervised methods for segmentation, with an emphasis on improving accuracy and reliability through feature extraction, misclassification cleanup, and thorough evaluation.

Subtask:

   a)Data Loader for Images:
     The goal is to create a robust data loading mechanism that can efficiently handle image data. This involves reading image files, preprocessing them (e.g., resizing, normalization), and preparing them for training a machine learning model.

  b)Binary Image Classification:
    Develop a deep learning model to classify images into two categories: images that contain cracks (positive class) and those that do not (negative class). The goal is to accurately distinguish between these two classes.

  c)Feature Extraction to Improve Model Performance:
    Extract meaningful features from the images to enhance the classification model's performance. This can involve techniques like using pre-trained models (transfer learning), applying convolutional layers to learn spatial features, or using traditional image processing methods to highlight relevant features.

  d)Segmentation Workflow for Positive Images:
    For images classified as containing cracks, develop a method to segment the cracks. This involves isolating the regions in the image that correspond to cracks, possibly using methods like thresholding, edge detection, or deep learning-based segmentation models.

Conclusion: ResNet-18 is generally the better choice for crack detection and segmentation, especially on larger and more complex datasets due to its superior feature extraction capabilities and ability to train very deep networks effectively.
            AlexNet is generally the better choice for crack detection and segmentation, especially on larger and more complex datasets due to its superior feature extraction capabilities and ability to train very deep networks effectively.


