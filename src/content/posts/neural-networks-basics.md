---
title: "Neural Networks Basics: From Perceptron to Multi-Layer Perceptron"
published: 2025-01-20
description: Understanding neural networks from the simplest perceptron to multi-layer perceptrons with Python and PyTorch implementations.
tags: [Neural Networks, Deep Learning, Python, PyTorch]
category: 技术
draft: false
---

## 1. What are Neural Networks?

Neural networks are computational models inspired by biological neurons, consisting of a large number of interconnected nodes (neurons). Each neuron receives inputs, performs weighted summation, and produces output through an activation function. Neural networks learn patterns in data by adjusting weights and biases.

## 2. Perceptron

The perceptron is the simplest neural network model, proposed by Frank Rosenblatt in 1957. It can only solve linearly separable problems.

### 2.1 Structure of Perceptron

A perceptron consists of:

- **Input Layer**: Receives feature vectors
- **Weights**: Each input has a weight
- **Bias**: A constant term
- **Activation Function**: Usually a step function or sign function

### 2.2 Mathematical Representation

For input vector `x = [x1, x2, ..., xn]`, the perceptron output is:

```
y = f(sum(wi * xi) + b)

where:
- wi is the weight of the i-th input
- b is the bias
- f is the activation function
```

### 2.3 Python实现感知机

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 初始化权重和偏置
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 训练过程
        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                # 计算预测值
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)

                # 更新权重和偏置
                update = self.learning_rate * (y[i] - prediction)
                self.weights += update * X[i]
                self.bias += update

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])
```

## 3. 多层感知机（MLP）

多层感知机是感知机的扩展，包含一个或多个隐藏层。这使得MLP能够学习非线性关系，解决更复杂的问题。

### 3.1 MLP的结构

- **输入层**：接收原始特征
- **隐藏层**：一个或多个中间层，进行特征变换
- **输出层**：产生最终预测

### 3.2 前向传播

前向传播是数据从输入层流向输出层的过程：

```python
def forward_propagation(X, weights, biases, activation_func):
    """
    X: 输入数据
    weights: 权重列表 [W1, W2, ..., Wn]
    biases: 偏置列表 [b1, b2, ..., bn]
    activation_func: 激活函数
    """
    activations = [X]

    for W, b in zip(weights, biases):
        z = np.dot(activations[-1], W) + b
        a = activation_func(z)
        activations.append(a)

    return activations
```

### 3.3 反向传播算法

反向传播是训练神经网络的核心算法，通过计算损失函数对权重的梯度来更新参数。

基本步骤：

1. 前向传播计算输出和损失
2. 计算输出层的误差
3. 将误差反向传播到隐藏层
4. 计算梯度并更新权重

### 3.4 使用PyTorch实现MLP

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 使用示例
model = MLP(input_size=784, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 4. 激活函数

激活函数引入非线性，使神经网络能够学习复杂模式。常用的激活函数包括：

- **Sigmoid**：`sigma(x) = 1 / (1 + e^(-x))`，输出范围(0,1)
- **Tanh**：`tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`，输出范围(-1,1)
- **ReLU**：`ReLU(x) = max(0, x)`，最常用的激活函数
- **Leaky ReLU**：解决ReLU的"死亡神经元"问题

## 5. 损失函数和优化器

### 5.1 常用损失函数

- **均方误差（MSE）**：用于回归问题
- **交叉熵损失**：用于分类问题
- **二元交叉熵**：用于二分类问题

### 5.2 优化器

- **随机梯度下降（SGD）**：基础优化器
- **Adam**：自适应学习率，最常用
- **RMSprop**：适合处理非平稳目标

## 6. 实践建议

- 从简单模型开始，逐步增加复杂度
- 使用合适的激活函数和初始化方法
- 注意过拟合问题，使用正则化技术
- 合理设置学习率和批次大小
- 使用验证集监控模型性能

## 7. 总结

神经网络是深度学习的基石。从简单的感知机到复杂的多层感知机，理解这些基础概念对于深入学习深度学习至关重要。通过实践和项目，你会逐渐掌握如何设计和训练有效的神经网络模型。

> **提示：** 建议使用现代深度学习框架（如PyTorch、TensorFlow）来实现神经网络，它们提供了自动微分、GPU加速等强大功能，可以大大提高开发效率。
