---
title: "Comparing Optimization Algorithms: Gradient Descent, Newton's Method, and BFGS"
published: 2025-11-24
description: A controlled comparison of gradient descent, Newton's method, and BFGS on logistic regression, analyzing convergence behaviour and collinearity effects.
tags: [Optimization, Machine Learning, Gradient Descent, BFGS, Logistic Regression]
category: 研究
draft: false
---

## Introduction

The aim of this research is to compare the behaviour of three optimisation algorithms: **gradient descent**, **Newton's method**, and **BFGS** within a controlled logistic regression setting.

For a feature vector $\mathbf{x} \in \mathbb{R}^p$, logistic regression models the positive-class probability via:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \mathbf{w}) = \frac{1}{1 + e^{-\mathbf{x}^\top \mathbf{w}}}$$

This leads to a linear decision boundary $\mathbf{x}^\top \mathbf{w} = 0$ in the two-dimensional case.

To provide a tractable environment for comparison, we construct a synthetic dataset from a known logistic model with true log-odds:

$$\text{logit}(P(y=1)) = 2x_1 - 3x_2$$

corresponding to the ground-truth parameter vector $\mathbf{w}^* = (2, -3)$.

In this research, **Part A** compares the convergence behaviour and the estimated decision boundaries obtained by gradient descent, Newton's method, and BFGS on the synthetic dataset. **Part B** investigates how near-collinearity in the design matrix affects curvature, conditioning, and the resulting performance and stability of these optimisation algorithms.

## Part A: Decision Boundaries and Convergence Analysis

### Decision Boundaries Comparison

To assess whether different optimisation algorithms are able to recover the true logistic regression classifier, we visualised and compared with the ground-truth separator defined by $\mathbf{w}^* = (2, -3)$.

It is worth mentioning that for gradient descent method, we use step size of 15, because for step size of 0.2, it wastes too many iterations. By adjusting the step sizes, we find that their boundaries almost overlapped and their coefficients are all identical.

#### Quantitative Comparison

| Method | $w_1$ | $w_2$ | Intercept | Slope | Angle | $\|\mathbf{w} - \mathbf{w}^*\|$ |
|---|---|---|---|---|---|---|
| **True boundary** | 2.000 | -3.000 | 0.000 | 0.6667 | 0.00 | 0.00 |
| Gradient Descent | 2.709 | -4.203 | 0.000 | 0.6445 | 0.89 | 1.3963 |
| Newton's Method | 2.710 | -4.205 | 0.000 | 0.6445 | 0.89 | 1.3991 |
| BFGS | 2.710 | -4.205 | 0.000 | 0.6445 | 0.89 | 1.3991 |

All methods yield slopes that are very close to the ground-truth value. The angular deviations are all below 1 degree, showing that the estimated boundaries are almost perfectly aligned with the true separator.

### Convergence Behaviour Analysis

As shown in the analysis, **Newton's Method** exhibits the fastest convergence, reaching the optimum in only eight iterations. **BFGS** converges slightly more slowly, requiring seventeen iterations, whereas fixed-step **Gradient Descent** is significantly slower: even with a favourable step size, it still takes roughly seventy iterations to reach a similar objective value.

These observations are consistent with the classical convergence rates:

- Newton's Method achieves **quadratic convergence**
- BFGS enjoys **superlinear convergence**
- Gradient Descent converges only **linearly**

#### Effect of Step Size on Gradient Descent

For fixed-step Gradient Descent, the choice of the step size has a substantial impact on the convergence behaviour. A small step size results in smooth but extremely slow progress, whereas a very large step size causes strong oscillations.

| Method | Step Size | Iterations | Final Objective | Final Gradient Norm |
|---|---|---|---|---|
| Gradient Descent | 0.2 | 3595 | 0.229718 | $2.50 \times 10^{-5}$ |
| Gradient Descent | 5 | 211 | 0.229718 | $9.89 \times 10^{-7}$ |
| Gradient Descent | 15 | 70 | 0.229718 | $3.24 \times 10^{-7}$ |
| Gradient Descent | 30 | 51 | 0.229718 | $1.30 \times 10^{-7}$ |
| Newton's Method | --- | 8 | 0.229718 | $4.01 \times 10^{-12}$ |
| BFGS | --- | 17 | 0.229718 | $4.93 \times 10^{-11}$ |

#### Optimization Trajectories in Parameter Space

The trajectories reveal several characteristic behaviours:

- **Gradient Descent (step size = 30):** The elongated level sets of the objective create strong curvature anisotropy, causing GD to follow a pronounced zigzag pattern. The large step size further amplifies this behaviour, leading to overshooting and oscillation near the minimiser.
- **BFGS:** The early iterations exhibit GD-like behaviour because the inverse Hessian approximation is initially crude. As more secant updates accumulate curvature information, the search directions become better scaled, and the trajectory rapidly transitions toward a Newton-like path.
- **Newton's Method:** Using the exact Hessian, Newton's Method produces an update direction that almost directly points to the minimiser from the very first step, demonstrating optimal rescaling and quadratic convergence.

### Superlinear Convergence of Newton's Method

To empirically verify superlinear convergence, we compare the error ratios $e_{k+1}/e_k$. A sequence is superlinear if:

$$\frac{e_{k+1}}{e_k} \longrightarrow 0$$

Gradient descent: the first five ratios are $(0.9854, 0.9858, 0.9863, 0.9867, 0.9871)$, which remain nearly constant -- linear convergence.

Newton's method: the first five error ratios are $(0.6343, 0.4562, 0.2306, 0.0568, 0.0033)$, which decrease rapidly toward zero -- quadratic convergence confirmed.

### Early-Stage Behaviour of BFGS

We compare BFGS's first three update directions with the Newton directions. The cosine similarities between BFGS and Newton directions:

| Iteration | Cosine Similarity |
|---|---|
| 0 | 0.9972 |
| 1 | 0.9968 |
| 2 | 0.9983 |

The cosine similarities exceed 0.996 in all three iterations, demonstrating that the early BFGS search directions are almost perfectly aligned with Newton directions. The BFGS update formula:

$$B_{k+1} = B_k + \frac{\mathbf{y}_k \mathbf{y}_k^\top}{\mathbf{y}_k^\top \mathbf{s}_k} - \frac{B_k \mathbf{s}_k \mathbf{s}_k^\top B_k}{\mathbf{s}_k^\top B_k \mathbf{s}_k}$$

rapidly captures the local geometry, explaining why BFGS quickly enters a superlinear convergence regime.

## Part B: Nearly Collinear Data and Its Effect on Convergence

### Gradient Descent Behaviour Under Near-Collinearity

We perturb the second feature as:

$$X_{\cdot,2} = X_{\cdot,1} + \varepsilon \xi, \qquad \xi \sim \mathcal{N}(0,1)$$

with $\varepsilon \in \{0.1, 0.01, 0.001, 0.0001\}$.

GD reaches the same objective value for all epsilon, indicating the statistical problem remains well-posed. However, the optimisation dynamics differ significantly:

- $\varepsilon = 0.1$: smooth and stable decay to $10^{-7}$
- $\varepsilon = 0.01$: an initial drop followed by a long plateau at $10^{-5}$ to $10^{-6}$
- $\varepsilon = 0.001$: an even longer plateau with almost no reduction in gradient norm
- $\varepsilon = 0.0001$: rapid convergence within only 23 iterations

GD slows dramatically when epsilon lies in the range $10^{-2}$ to $10^{-3}$, but becomes fast again when epsilon is extremely small. At $\varepsilon = 0.0001$, the Hessian nearly collapses into a rank-one structure, effectively reducing the optimisation to one dimension.

### Newton's Method Behaviour Under Near-Collinearity

Newton's method is remarkably robust to moderate levels of feature collinearity. As epsilon decreases from $10^{-1}$ to $10^{-4}$, the conditioning deteriorates by several orders of magnitude, yet the algorithm maintains essentially identical objective trajectories and converges in roughly six iterations for all tested values.

### Key Findings

1. **Gradient Descent** is most severely affected by near-collinearity, with slowdown being worst not when collinearity is strongest, but when the Hessian is maximally anisotropic.
2. **Newton's Method** achieves quadratic convergence regardless of collinearity level, thanks to exact curvature information from the Hessian.
3. **BFGS** bridges the gap between GD and Newton's method, quickly accumulating curvature information to achieve superlinear convergence.
4. The interplay between curvature anisotropy and optimization algorithm choice is critical for practical machine learning applications.
