---
title: "Simulation-Based Inference: Neural Posterior Estimation"
published: 2025-01-16
description: SBI方法介绍，使用神经网络直接估计后验分布，涵盖NPE原理、条件归一化流和实际应用。
tags: [SBI, Bayesian Inference, Normalising Flows, PyTorch]
category: 研究
draft: false
---

## 1. 什么是Simulation-Based Inference？

Simulation-Based Inference (SBI)，也称为Likelihood-Free Inference，是一类用于从观测数据推断模型参数的方法。当模型的似然函数难以计算或不可用时，SBI方法特别有用。

传统的贝叶斯推理需要计算似然函数 `p(x|theta)`，但在许多复杂模型中（如生物系统、物理模拟等），这个似然函数是难以解析计算的。SBI通过模拟来解决这个问题。

## 2. 为什么需要SBI？

许多科学和工程问题涉及复杂的模拟模型：

- **计算生物学**：基因调控网络、蛋白质折叠
- **天体物理学**：宇宙学模拟、引力波分析
- **流行病学**：疾病传播模型
- **神经科学**：神经元动力学模型

这些模型通常只能通过模拟来评估，无法直接计算似然函数。

## 3. SBI的基本框架

SBI的核心思想是：

1. 从先验分布 `p(theta)` 中采样参数
2. 使用这些参数运行模拟，生成数据 `x ~ p(x|theta)`
3. 学习从观测数据到参数的后验映射

## 4. Neural Posterior Estimation (NPE)

Neural Posterior Estimation是SBI的一种方法，使用神经网络直接估计参数的后验分布 `p(theta|x)`。

### 4.1 NPE的基本原理

NPE训练一个神经网络 `q_phi(theta|x)` 来近似真实的后验分布 `p(theta|x)`。训练目标是最大化：

```
E_{p(theta,x)}[log q_phi(theta|x)]

其中 p(theta,x) = p(theta) p(x|theta)
```

### 4.2 训练过程

```python
import torch
import torch.nn as nn
import numpy as np

class PosteriorEstimator(nn.Module):
    def __init__(self, data_dim, param_dim, hidden_dim=128):
        super(PosteriorEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim * 2)  # 均值和方差
        )

    def forward(self, x):
        output = self.net(x)
        mean = output[:, :self.param_dim]
        log_std = output[:, self.param_dim:]
        return mean, log_std

    def sample(self, x, n_samples=1000):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        samples = torch.normal(mean, std)
        return samples

def train_npe(simulator, prior, n_simulations=10000):
    # 生成训练数据
    params = prior.sample((n_simulations,))
    data = simulator(params)

    # 初始化模型
    model = PosteriorEstimator(data_dim=data.shape[1],
                              param_dim=params.shape[1])
    optimizer = torch.optim.Adam(model.parameters())

    # 训练循环
    for epoch in range(1000):
        # 随机打乱数据
        indices = torch.randperm(n_simulations)
        params_shuffled = params[indices]
        data_shuffled = data[indices]

        # 前向传播
        mean, log_std = model(data_shuffled)
        std = torch.exp(log_std)

        # 计算负对数似然
        dist = torch.distributions.Normal(mean, std)
        loss = -dist.log_prob(params_shuffled).mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
```

## 5. 条件归一化流（Conditional Normalizing Flows）

更高级的NPE方法使用条件归一化流来建模复杂的后验分布。归一化流能够表示任意复杂的概率分布。

### 5.1 归一化流基础

归一化流通过一系列可逆变换将一个简单分布（如标准正态分布）转换为复杂分布：

```python
class ConditionalFlow(nn.Module):
    def __init__(self, data_dim, param_dim, n_transforms=5):
        super(ConditionalFlow, self).__init__()
        self.transforms = nn.ModuleList([
            AffineCouplingLayer(data_dim, param_dim)
            for _ in range(n_transforms)
        ])

    def forward(self, params, data):
        log_det = 0
        z = params
        for transform in self.transforms:
            z, ld = transform(z, data)
            log_det += ld
        return z, log_det

    def inverse(self, z, data):
        log_det = 0
        params = z
        for transform in reversed(self.transforms):
            params, ld = transform.inverse(params, data)
            log_det += ld
        return params, log_det
```

## 6. 实际应用示例

### 6.1 简单示例：高斯模型

假设我们有一个简单的模型：观测数据来自高斯分布，我们想推断均值和标准差。

```python
def gaussian_simulator(params, n_obs=100):
    """模拟器：从高斯分布生成数据"""
    mean, std = params[:, 0], params[:, 1]
    data = []
    for m, s in zip(mean, std):
        samples = np.random.normal(m, s, n_obs)
        # 使用摘要统计量
        summary = np.array([np.mean(samples), np.std(samples)])
        data.append(summary)
    return np.array(data)

# 定义先验
from scipy.stats import uniform, gamma
prior_mean = uniform(loc=-5, scale=10)  # 均值在[-5, 5]
prior_std = gamma(a=2, scale=1)  # 标准差

# 训练NPE模型
model = train_npe(gaussian_simulator, prior, n_simulations=50000)

# 对新观测数据进行推理
observed_data = np.array([[2.5, 1.2]])  # 观测到的均值和标准差
posterior_samples = model.sample(torch.tensor(observed_data, dtype=torch.float32))
print(f"后验均值估计: {posterior_samples.mean(axis=0)}")
```

## 7. 评估和验证

评估NPE模型质量的方法：

- **后验预测检查**：从后验采样参数，生成模拟数据，检查是否与观测数据一致
- **校准检查**：检查后验分位数的校准
- **覆盖率**：检查真实参数是否落在后验置信区间内

## 8. 挑战和解决方案

### 8.1 高维数据

当数据维度很高时，直接使用原始数据可能不高效。解决方案是使用摘要统计量（Summary Statistics）或学习摘要网络。

### 8.2 模拟成本

如果模拟非常耗时，可以使用：

- 主动学习：选择最有信息量的参数进行模拟
- 模拟器替代：训练神经网络替代昂贵的模拟器

## 9. 研究展望

SBI和NPE是一个快速发展的领域，未来的研究方向包括：

- 更高效的归一化流架构
- 处理高维参数空间的方法
- 结合领域知识的约束
- 不确定性量化

## 10. 总结

Simulation-Based Inference和Neural Posterior Estimation为处理复杂模型提供了强大的工具。当传统方法无法应用时，SBI使我们能够从观测数据中推断模型参数，这对于许多科学和工程问题至关重要。

在我的EPSRC资助项目中，我们正在探索这些方法在统计建模和机器学习中的应用，希望能为这一领域做出贡献。

> **推荐资源：**
> - sbi library: Python包，提供了SBI的完整实现
> - Papamakarios et al. (2019): "Neural Likelihood Estimation"
> - Greenberg et al. (2019): "Automatic Posterior Transformation for Likelihood-Free Inference"
