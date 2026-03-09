# ASL-CNN-Classifier

A Deep Learning project using **PyTorch** to classify hand gestures from the American Sign Language (ASL) dataset. The project implements a multi-layer Convolutional Neural Network (CNN) to achieve high-accuracy image recognition on the `sign_mnist` dataset.

## Project Overview

This repository contains a complete pipeline for training a computer vision model to recognize 24 distinct letters of the ASL alphabet. The model is trained on $28 \times 28$ grayscale images and is designed to serve as a backbone for real-time sign language interpretation.



## Technical Architecture

The model utilizes a custom CNN architecture optimized for small-scale image classification:

* **Input Layer:** $28 \times 28$ grayscale images.
* **Convolutional Layers:** * **Layer 1:** 32 filters, $3 \times 3$ kernel, ReLU activation.
    * **Layer 2:** 64 filters, $3 \times 3$ kernel, ReLU activation.
    * **Layer 3:** 128 filters, $3 \times 3$ kernel, ReLU activation.
* **Pooling:** Max-Pooling layers $(2 \times 2)$ following each convolution to reduce spatial dimensions.
* **Fully Connected Layers:** A dense layer with 512 units followed by a Softmax output layer for the 24 classes.

## Performance & Insights

| Metric | Accuracy | Loss |
| :--- | :--- | :--- |
| **Training** | 100% | ~0.0001 |
| **Validation** | 93.1% | ~0.4716 |

### Analysis of Results
During the training of 10 epochs, the model reached perfect accuracy on the training set. However, a divergence between training and validation loss suggests **overfitting**. The model has begun to memorize specific noise in the training data rather than purely generalizable patterns.

## Future Enhancements

* **Regularization:** Introduce Dropout layers and Weight Decay (L2) to mitigate overfitting.
* **Data Augmentation:** Implement random rotations, scaling, and shearing to improve model robustness.
* **Live Inference:** Integrate with **OpenCV** to perform real-time hand tracking and classification via webcam.
* **Mobile Deployment:** Convert the PyTorch model to TorchScript or ONNX for edge device usage.

## Setup & Requirements

1.  **Environment:** Python 3.x
2.  **Dependencies:**
    ```bash
    pip install torch pandas numpy matplotlib scikit-learn
    ```
3.  **Data:** Ensure `sign_mnist_train.csv` and `sign_mnist_valid.csv` are in the project root.

---
*Developed as part of a Computer Vision and Deep Learning exploration.*
