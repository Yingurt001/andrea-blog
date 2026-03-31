---
title: "Computer Vision Lesson 1: Image Classification with Transfer Learning"
published: 2025-01-22
description: Learning image classification using VGG16 transfer learning with TensorFlow/Keras.
tags: [Computer Vision, Deep Learning, Transfer Learning, TensorFlow]
category: 技术
draft: false
---

## Learning Topics

### 1. Data Loading and Preprocessing

- Using `image_dataset_from_directory` to load image datasets
- Image size standardization (128x128)
- Data type conversion (to float32)
- Data caching and prefetching optimization (using AUTOTUNE)

### 2. Using VGG16 Pre-trained Model

- Importing VGG16 pre-trained model
- Understanding the role of `include_top=False`
- Setting `input_shape=(128, 128, 3)`
- Freezing pre-trained layers (`trainable = False`)

### 3. Transfer Learning Practice

- Building transfer learning model architecture
- Adding custom layers on top of pre-trained base
- Using Flatten layer to flatten features
- Adding fully connected layers for classification

### 4. Model Training and Evaluation

- Model compilation (using Adam optimizer and binary_crossentropy loss function)
- Training process monitoring (30 epochs)
- Validation set evaluation
- Training history visualization

### 5. Model Architecture Design

- Sequential model construction
- Importance of layer stacking order
- Activation function selection (ReLU and Sigmoid)
- Output layer design (binary classification problem)

### 6. Performance Optimization Techniques

- Data pipeline optimization (cache and prefetch)
- Batch size setting (batch_size=64)
- Random seed setting for reproducibility
- Training and validation data separation

## Learning Outcomes

Through this lesson, I have mastered:

- How to use TensorFlow/Keras to load image datasets
- How to leverage pre-trained VGG16 model for transfer learning
- How to build and train an image classification model
- How to evaluate model performance and visualize training process

## Related Files

The complete code implementation can be found in the repository:

- [Lecture_1.ipynb](https://github.com/Yingurt001/Computer-Vision/blob/main/Lecture_1.ipynb) - Complete code implementation
- [Lesson_1.md](https://github.com/Yingurt001/Computer-Vision/blob/main/Lesson_1.md) - Lesson notes

> **Key Takeaway:** Transfer learning allows us to leverage pre-trained models like VGG16 to achieve good performance on new tasks with limited data and computational resources. By freezing the pre-trained layers and only training the top layers, we can quickly adapt the model to our specific classification problem.

**Repository:** [https://github.com/Yingurt001/Computer-Vision](https://github.com/Yingurt001/Computer-Vision)
