---
title: Supervised Machine Learning Basics
published: 2025-01-22
description: Fundamental concepts of supervised learning -- loss functions, empirical risk minimization, softmax, logistic regression, and MLE.
tags: [Machine Learning, Statistics, Classification, Regression]
category: 技术
draft: false
---

## Target of Supervised Machine Learning

In supervised machine learning, we have a dataset that contains both target variable ($y$) and feature variables ($\mathbf{X}$). The goal is to learn a function that maps inputs to outputs based on example input-output pairs.

Examples include:

- **Linear Regression**: Predicting continuous values
- **Binary Classification**: Classifying into two classes
- **Multi-class Classification**: Classifying into multiple classes
- **Multi-label Classification**: Assigning multiple labels to each instance

*Note:* In statistics, we typically call $y$ the **dependent variable** and $\mathbf{X}$ the **independent variables** (or features).

## Example: Binary Classification

For binary classification, supervised machine learning means we want a classifier that can reliably give us the correct labels for any input.

### How to Measure Performance?

> We use a **misclassification rate** to measure performance.

The loss function for misclassification rate is:

$$\mathcal{L}(\theta) \triangleq \frac{1}{N}\sum_{n=1}^{N}\mathbb{I}(y_n \neq f(\boldsymbol{x}_n;\theta))$$

where $\mathbb{I}(\cdot)$ is the indicator function that returns 1 if the condition is true, and 0 otherwise.

## Model Fitting: Empirical Risk Minimization

Model fitting is done by minimizing the loss function to find the optimal $\theta$, which is also called **empirical risk minimization**.

$$\hat{\theta} = \arg\min_{\theta} \mathcal{L}(\mathbf{\theta}) = \arg\min_\theta\frac{1}{N}\sum_{n=1}^{N} \ell \bigl(y_n, f(\mathbf{x}_n; \theta) \bigr)$$

This optimization process finds the parameters that minimize the average loss over the training data.

## Uncertainty in Predictions

### Why Do We Need Uncertainty?

We should avoid false confidence. There are two sources from which we get uncertainty:

- **Epistemic uncertainty:** Arises because of lack of knowledge (model uncertainty)
- **Aleatoric uncertainty:** Due to the randomness in data (data uncertainty)

Sometimes, the model cannot give us the exact answer to our prediction, and we need to use conditional probability distribution to describe the uncertainty:

$$p(y=c|\mathbf{x};\mathbf{\theta})=f_c(\mathbf{x};\mathbf{\theta})$$

### Softmax Function

We use raw model scores (any real numbers) which are the direct output of our model. Theoretically, we need the function $f_c(\mathbf{x};\mathbf{\theta})$ to be within the range $(0,1)$. To avoid this restriction, we can let the model return unnormalized log-probabilities. We can then convert these to probabilities using the **softmax function**:

$$\mathcal{S}(\mathbf{a}) \triangleq \left[\frac{e^{a_1}}{\sum_{c'=1}^{C}e^{a_{c'}}},\cdots, \frac{e^{a_C}}{\sum_{c'=1}^{C}e^{a_{c'}}} \right]$$

We use exponentials because:

- Exponentials are always positive
- It is easy to normalize
- Larger scores have exponentially larger influence

Hence the final probability becomes:

$$p(y=c|\mathbf{x};\theta)=\mathcal{S}(f(\mathbf{x};\theta))_c$$

## Understanding Logistic Regression

In statistics, Logistic Regression is similar to linear regression, but for classification. We write it as:

$$f(\mathbf{x};\mathbf{w})=\mathbf{w}^T\mathbf{x}$$

However, for binary classification, we apply the sigmoid function to get probabilities:

$$p(y=1|\mathbf{x};\mathbf{w}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

## Maximum Likelihood Estimation (MLE)

When fitting probabilistic models, we often use **negative log likelihood** as the loss function:

$$\ell(y,f(\mathbf{x};\mathbf{\theta}))=-\log p(y \mid f(\mathbf{x};\mathbf{\theta}))$$

### Why Negative Log Likelihood?

A good model should assign a high probability to the true output $y$ for each corresponding input $\mathbf{x}$. The average negative log probability of the training set is:

$$\text{NLL}(\mathbf{\theta}) = -\frac{1}{N}\sum_{n=1}^{N}\log p(y_n\mid f(\mathbf{x}_n;\mathbf{\theta}))$$

In statistics, if we minimize this, we can compute the **maximum likelihood estimate** or MLE:

$$\hat{\theta}_{\text{MLE}}=\arg\min_\mathbf{\theta}\text{NLL}(\mathbf{\theta})$$

## Implementation of MLE in Linear Regression

For linear regression, the MLE solution has a closed-form expression. Given the model:

$$y = \mathbf{w}^T\mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

The MLE estimate for the weights is:

$$\hat{\mathbf{w}}_{\text{MLE}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

where $\mathbf{X}$ is the design matrix (each row is a data point) and $\mathbf{y}$ is the target vector.

## Summary

This article introduced the fundamental concepts of supervised machine learning:

- Understanding the target of supervised learning (regression and classification)
- Measuring performance using loss functions
- Model fitting through empirical risk minimization
- Handling uncertainty using probability distributions
- The softmax function for multi-class classification
- Maximum likelihood estimation as a principled approach to model fitting

These concepts form the foundation for understanding more advanced machine learning techniques.

> **Next Steps:** In future posts, we'll explore specific algorithms like logistic regression, neural networks, and deep learning models, building upon these fundamental concepts.
