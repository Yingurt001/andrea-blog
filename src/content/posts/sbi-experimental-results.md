---
title: "SBI Experimental Report: From Toy Models to Epidemic Inference"
published: 2026-03-21
description: Complete experimental journey of Neural Posterior Estimation -- from Poisson-Gamma validation to SIR epidemic inference with normalising flows.
tags: [SBI, Normalising Flows, Bayesian Inference, NPE, PyTorch]
category: 研究
draft: false
---

## Overview

This post documents the complete experimental journey of my EPSRC-funded Simulation-Based Inference (SBI) project. The core question: **when the likelihood function of a statistical model is intractable, how can we still perform Bayesian parameter inference?**

The answer is Neural Posterior Estimation (NPE): sample parameters from the prior, run the simulator to generate synthetic data, and train a neural network $q_\phi(\theta|y)$ to directly approximate the posterior distribution. Once trained, the network can instantly output posteriors for any new observation (amortised inference).

I compare three NPE architectures of increasing expressiveness:

| Method | Posterior Representation | Expressiveness |
|---|---|---|
| Single Gaussian (SG) | $\mathcal{N}(\mu_\phi(y), \sigma^2_\phi(y))$ | Low: symmetric unimodal only |
| Mixture of Gaussians (MoG, K=8) | $\sum_k \pi_k \mathcal{N}(\mu_k, \sigma^2_k)$ | Medium: multimodal, asymmetric |
| Normalising Flow (NF, 5 layers) | Affine coupling + LeakyReLU | High: arbitrarily complex |

All models are built from scratch in PyTorch with a unified training objective: $\max_\phi \sum_i \log q_\phi(\theta^{(i)} | y^{(i)})$.

## Experiment 1: Poisson-Gamma Toy Example (Fixed n=100)

**Goal:** Validate all three NPE methods on a model with a known analytical posterior (ground truth).

**Model:** Prior $\theta \sim \text{Gamma}(2, 3)$; Simulator $Y_1, \ldots, Y_{100} \mid \theta \sim \text{Poisson}(\theta)$

**Sufficient statistic:** $y_{\text{sum}} = \sum Y_i$

**True posterior (conjugacy):** $\theta \mid y_{\text{sum}} \sim \text{Gamma}(2 + y_{\text{sum}}, 103)$

**Training data:** $10^6$ simulated $(\theta, y_{\text{sum}})$ pairs

### Results

| Model | Final Training Loss |
|---|---|
| Single Gaussian | -1.20 |
| MoG (K=8) | **-1.23** |
| NF (5 layers) | -1.18 |

**Takeaway:** All three methods produce reasonable posterior approximations on this simple toy example. MoG slightly outperforms SG. The NF v1 had an initialisation issue (LeakyReLU slope fixed at 0.3 caused initial loss to explode to 1505), which was fixed by making the slope a learnable parameter.

## Experiment 2: Flexible N -- Generalisation Across Sample Sizes

**Goal:** Can the model generalise to sample sizes $n$ never seen during training?

**Training:** $n \in \{5, 10, 15, \ldots, 50\}$

**Testing (interpolation):** $n = 7, 35$ -- within training range but unseen

**Testing (extrapolation):** $n = 75, 150$ -- beyond training range

### The Problem: Standard NF Collapses at Large n

The standard NF uses $\log(n)/\log(50)$ as an input feature. At $n=150$, this equals 1.28, **outside the training range** $[0, 1]$. The conditioner network produces unpredictable flow parameters. Additionally, the true posterior at large $n$ is extremely narrow ($\sigma_n \sim O(1/\sqrt{n})$), requiring the flow to compress $\mathcal{N}(0,1)$ by a factor of ~50, which is numerically unstable.

### The Solution: CLT Reparameterisation

Instead of training the flow in $\theta$-space, I train it in **CLT-standardised space**:

$$z_{\text{CLT}} = \frac{\theta - \hat{\mu}_n}{\hat{\sigma}_n}, \quad \hat{\mu}_n = y/n, \quad \hat{\sigma}_n = \sqrt{y/n}/\sqrt{n}$$

By the Central Limit Theorem, $z_{\text{CLT}} \approx \mathcal{N}(0,1)$ for all $n$. The flow only needs to learn a **small correction** from the CLT approximation to the true posterior.

**Intuition:** CLT handles the "heavy lifting" (locating position and width), the flow handles the "fine-tuning" (correcting shape and skewness).

**Takeaway:** CLT reparameterisation solves the extrapolation failure by exploiting known asymptotic mathematical structure (CLT) to anchor the neural network, reducing learning difficulty and improving generalisation.

## Experiment 3: Flow Architecture Comparison -- Affine vs. Neural Spline

**Goal:** Diagnose and address the limited expressiveness of affine flows.

The affine + LeakyReLU flow is fundamentally a **piecewise-linear function**. The fix: replace each layer with **monotone rational-quadratic splines** (Neural Spline Flow, NSF).

| Model | Per-layer expressiveness | Total params (5 layers) |
|---|---|---|
| CLT-Affine-5L | 2-segment piecewise linear | ~18k |
| CLT-Affine-15L | Finer piecewise linear | ~19k |
| **CLT-NSF (K=8)** | **Smooth curve (8-segment spline)** | **~35k** |

### Debugging: NSF Training Failure (Loss stuck at 100,000)

First attempt with CLT-NSF: loss started at 100,902 and barely moved after 1000 epochs. Root cause: when $y_{\text{sum}} = 0$ (a valid Poisson outcome), the CLT standardisation produces $z_{\text{CLT}} \approx 15{,}000$, far beyond the spline's working range $[-B, B]$.

Fix: clamp CLT statistics to sensible minimums ($\hat{\mu} \geq 0.05$, $\hat{\sigma} \geq 0.02$), expand spline range to $B=6$, and replace `torch.where` with a multiplication mask to prevent NaN gradient leakage.

**Takeaway:** NSF is theoretically superior (smooth transforms vs. piecewise-linear), but practically more fragile at boundary cases. The debugging process revealed important numerical pitfalls in CLT standardisation and PyTorch's gradient-unsafe `torch.where`.

## Experiment 4: SIR Epidemic Model -- Truly Intractable Likelihood

The SIR model is the first **genuinely likelihood-intractable** problem: the stochastic SIR simulator has exponentially many possible trajectories, making $p(R_\infty | \beta, \gamma)$ impossible to compute. This is precisely where SBI shines.

### Stage 1: Single-Parameter Inference

**Model:** Discrete-time stochastic SIR (Binomial transitions), $N=1000$, $I_0=1$

**Fixed:** Recovery rate $\gamma = 0.05$

**Infer:** Infection rate $\beta \sim U(0.05, 0.5)$

**Observation:** Final epidemic size $R_\infty$ (scalar)

**Training:** 200,000 simulations

**Baseline:** ABC-Rejection ($\epsilon=10$, up to $2 \times 10^6$ simulations)

### Quantitative Comparison

Wasserstein distance between NPE and ABC posteriors (lower is better):

| $R_\infty$ | SG | MoG | NF | Best |
|---|---|---|---|---|
| 100 | 0.0089 | 0.0018 | **0.0015** | NF |
| 300 | 0.0031 | **0.0005** | 0.0012 | MoG |
| 500 | 0.0018 | **0.0004** | 0.0005 | MoG |
| 700 | 0.0018 | 0.0012 | **0.0004** | NF |
| 900 | 0.0105 | 0.0047 | **0.0002** | NF |

### Speed Comparison: NPE vs ABC

- **NPE (Normalising Flow):** 0.2 ms per posterior
- **ABC-Rejection:** 13.6 s per posterior (78,000x slower)

### Stage 1 Conclusions

1. **NF excels when posteriors are skewed** ($R_\infty = 100, 900$), where its flexible nonlinear transforms outperform parametric families.
2. **MoG is a robust middle ground**, matching NF when posteriors are near-symmetric.
3. **SG is consistently worst**, especially at boundary-truncated posteriors.
4. **NPE is ~78,000x faster than ABC**, demonstrating the practical value of amortised inference.

### Stage 2: Joint Inference of $(\beta, \gamma)$ -- Identifiability

Moving to 2D: simultaneously infer both the infection rate $\beta$ and recovery rate $\gamma$, using Real NVP (2D coupling flow, 6 layers).

**Key finding -- the identifiability problem:** The final epidemic size $R_\infty$ primarily depends on $R_0 = \beta/\gamma$, not on $\beta$ and $\gamma$ individually. The 2D posterior forms a **diagonal band** along $\beta/\gamma = \text{const}$.

- **SG:** Can only fit an ellipse, worst performance.
- **MoG (K=5):** Reasonable but jagged at boundaries.
- **Real NVP:** Smoothly tracks the ABC scatter band. Best overall (loss: -4.72 vs MoG -4.66 vs SG -4.38).

Conclusion: To break the identifiability between $\beta$ and $\gamma$, we need richer observations (e.g., the full infection time series $I(t)$ rather than just $R_\infty$).

## Summary: Three Levels of Progression

| Level | Experiment | Core Question | Key Finding |
|---|---|---|---|
| **Validate** | Poisson-Gamma (fixed n) | Can NPE approximate known posteriors? | All three methods work; MoG >= NF > SG |
| **Improve** | Flexible N + CLT | How to extrapolate to unseen n? | CLT reparameterisation exploits math structure to anchor the network |
| **Apply** | SIR epidemic model | Truly intractable inference | NF best for skewed posteriors; NPE 78,000x faster than ABC |

### Core Insights Across All Experiments

1. **Expressiveness hierarchy:** SG < MoG < NF. Greater expressiveness yields larger gains on complex posteriors.
2. **Math structure + neural networks:** The success of CLT regularisation shows that explicitly encoding known mathematical structure (sufficient statistics, asymptotic theory) into the network architecture is more efficient and stable than learning from scratch.
3. **Amortised inference matters:** Train once, query instantly for any new observation. This is transformative for scenarios requiring repeated inference (e.g., real-time epidemic monitoring).

> **Code:** All experiments are available at [github.com/Yingurt001/SBI-summer-project](https://github.com/Yingurt001/SBI-summer-project). Built entirely from scratch in PyTorch -- no external SBI library dependencies.
