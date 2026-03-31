---
title: "Computer Vision Lesson 2: Understanding the Convolutional Base"
published: 2025-11-24
description: Deep dive into CNN fundamentals - convolution kernels, feature extraction, and ReLU activation.
tags: [Computer Vision, CNN, Deep Learning, TensorFlow]
category: 技术
draft: false
---

## Introduction

In Lesson 1, we learned that our convolutional classifier has two parts:

1. **A convolutional base** - extracts visual features from an image
2. **A head of dense layers** - uses those features to classify the image

In this lesson, we focus on understanding what the convolutional base is doing during CNN. We don't need to go into details about both parts, but we will try our best to understand the convolutional base.

## What Do We Learn in This Lesson?

We only learn one thing:

- **Convolutional layer with ReLU activation**

> **Important Note:** A convolutional layer is *not* equivalent to a filter. Our filter often does not include activation, but is just a kernel scanning that outputs the value for each grid cell.

## Feature Extraction

We first need to discuss the purpose of these layers in the network. Feature extraction consists of three basic operations:

1. **Filter** an image for a particular feature (convolution)
2. **Detect** that feature within the filtered image (ReLU)
3. **Condense** the image to enhance the features (maximum pooling)

## First Layer: Filter with Convolution

A convolutional layer carries out the filtering step.

### Weights and Kernels

The first figure shows what we call **weights**, which is what the convnet learns during training. These are primarily contained in its convolutional layers. These weights we call **kernels**. We can represent them as small arrays.

This figure shows a **3x3 convolution kernel** (filter). The weight values are:

```
[-1   2   -1]
[-1   2   -1]
[-1   2   -1]
```

### The Convolution Operation

The convolution operation does:

```
output = sum(kernel[i,j] * input_patch[i,j])
```

The kernel slides across the image, and at each location it produces a single output value. These values assemble into a feature map that highlights the features the kernel is designed to detect (e.g., vertical edges).

A kernel operates by scanning over an image and producing a weighted sum of pixel values. In this way, a kernel will act sort of like a polarized lens, emphasizing or deemphasizing certain patterns of information.

### Implementing Convolution in TensorFlow

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # More layers follow
])

'''
keras.Sequential means that I am building a model by stacking layers one after another

layers gives me the access to Keras layer types (Conv2D)

Conv2D is a 2D convolution layer used for images
'''
```

### Understanding the Parameters

**Filter = 64** means that the layer will learn 64 different feature maps.

If I set `filters=64`, then:

- I have 64 different kernels
- Each kernel learns a different pattern
- Each kernel shares weights across space

Because some features like edges, textures, or corners can appear anywhere in the image, they should be detected in the same way like other pixels.

Normally we would set the size of the kernel to be odd like `kernel_size=(3, 3)` or `(5, 5)` so that a single pixel sits at the center.

### More Fun Facts About Kernel

Each output neuron is related to nine neurons in the input. By setting the dimensions of the kernels with `kernel_size`, we will be able to tell the convnet how to form the connections.

## Second Layer: Activations

After the convolution layer produces its feature maps, the activation layer (like ReLU) applies a nonlinear function to those maps.

### Why Do We Need Activation?

Because without activation, a CNN would be only linear operations, meaning it cannot learn complex patterns. Activation makes the network capable of learning nonlinear features, such as shapes, textures, and objects.

### What the Activation Does

- Keeps positive values (ReLU: max(0, x))
- Suppresses irrelevant negative activations
- **Adds nonlinearity**

### How They Work Together

**Convolution layer produces:**
- Raw feature maps (linear responses)

**Activation layer produces:**
- Activated feature maps

### ReLU Activation

ReLU activation function:

- If **x < 0**, the output is **0**
- If **x >= 0**, the output is **x**

We would also call the rectifier function the **ReLU activation** or even the **ReLU function**. It can be defined in its own Activation layer, but most often we would also include it as the activation function of Conv2D.

After a convolution produces a feature map (linear output), ReLU is applied:

```
Convolution -> Feature Map -> ReLU -> Activated Feature Map
```

### Why Is This Important?

We could think about the activation function as scoring pixel values according to some measure of importance. Every unimportant value is equally unimportant.

## Summary

**Key Takeaway:** The convolutional base performs feature extraction through three steps: **Filter** (convolution), **Detect** (ReLU activation), and **Condense** (max pooling). Understanding how convolution kernels scan images and how ReLU activation adds nonlinearity is crucial for building effective CNNs.

## Learning Outcomes

Through this lesson, I have mastered:

- Understanding the structure of convolutional classifiers (base + head)
- How convolutional layers perform filtering through kernel scanning
- The relationship between kernels, weights, and feature maps
- How to use Conv2D layers in TensorFlow/Keras
- The importance of ReLU activation for adding nonlinearity
- How convolution and activation work together in feature extraction

## Related Files

- [lecture2.ipynb](https://github.com/Yingurt001/Computer-Vision/blob/main/lecture2.ipynb) - Complete code implementation
- [Lesson_2.md](https://github.com/Yingurt001/Computer-Vision/blob/main/Lesson_2.md) - Lesson notes
- [CNN_Lesson_2.md](https://github.com/Yingurt001/Computer-Vision/blob/main/CNN_Lesson_2.md) - Detailed lesson summary

> **Key Insight:** The convolutional base is the feature extraction engine of CNNs. By understanding how kernels scan images and how ReLU activation introduces nonlinearity, we can better design and optimize convolutional neural networks for various computer vision tasks.

**Repository:** [https://github.com/Yingurt001/Computer-Vision](https://github.com/Yingurt001/Computer-Vision)
