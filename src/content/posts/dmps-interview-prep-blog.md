---
title: 从 SBI 到 DMPS：生成模型、后验采样与我的暑研面试
published: 2026-04-01
description: 一篇长文数学科普——从 Normalizing Flow 到 Diffusion Model 到 Flow Matching，再到 DMPS 如何用闭式 likelihood score 做快速后验采样，最后是我把 NPE 和 DMPS 接起来的 Hybrid 实验。
tags:
  - SBI
  - DMPS
  - Flow Matching
  - 暑研
  - 面试
category: 科研
draft: false
---

这篇文章是我准备 Xiangming(ZJU) 暑研面试过程中的学习记录。从 normalizing flow 到 diffusion model 到 flow matching 再到 DMPS 的整条数学线索讲清楚

---

## 第一章：生成模型的三代演进

生成模型的目标只有一个：给定一堆数据样本 $\{\mathbf{x}^{(i)}\}_{i=1}^N$（比如人脸图片），学会从同一个分布 $p(\mathbf{x})$ 中**生成新样本**。三代方法用了三种完全不同的策略。

### 1.1 Normalizing Flow：可逆变换

#### 核心思想

Normalizing Flow 的想法最直接：学一个**可逆函数** $f_\theta$，把简单分布（标准高斯）变成数据分布。

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \xrightarrow{f_\theta} \quad \mathbf{x} = f_\theta(\mathbf{z}) \sim p_\theta(\mathbf{x})$$

因为 $f_\theta$ 可逆，反过来也成立：$\mathbf{z} = f_\theta^{-1}(\mathbf{x})$。这意味着我们可以**精确计算**任意一个点 $\mathbf{x}$ 的概率密度。

#### Change of Variables 公式

这是 normalizing flow 的数学基石。如果 $\mathbf{z}$ 的密度是 $p_Z(\mathbf{z})$，经过可逆变换 $\mathbf{x} = f(\mathbf{z})$ 后，$\mathbf{x}$ 的密度是：

$$p_X(\mathbf{x}) = p_Z\!\bigl(f^{-1}(\mathbf{x})\bigr) \cdot \left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|$$

取 log：

$$\log p_X(\mathbf{x}) = \log p_Z\!\bigl(f^{-1}(\mathbf{x})\bigr) + \log\left|\det \mathbf{J}_{f^{-1}}(\mathbf{x})\right|$$

其中 $\mathbf{J}_{f^{-1}}$ 是逆变换的 Jacobian 矩阵。

**为什么这很重要？** 因为有了精确的 $\log p_X(\mathbf{x})$，训练就是简单的最大似然：

$$\max_\theta \sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)})$$

不需要对抗训练（GAN），不需要变分下界（VAE），直接最大化似然。

#### 架构约束：可逆性的代价

问题来了：要让 $f_\theta$ 可逆且 Jacobian 行列式好算，架构设计非常受限。

最常见的做法是 **coupling layer**（如 RealNVP）：把输入 $\mathbf{z}$ 分成两半 $(\mathbf{z}_1, \mathbf{z}_2)$，其中一半保持不动，另一半做仿射变换：

$$\mathbf{x}_1 = \mathbf{z}_1$$
$$\mathbf{x}_2 = \mathbf{z}_2 \odot \exp\!\bigl(s(\mathbf{z}_1)\bigr) + t(\mathbf{z}_1)$$

其中 $s(\cdot)$ 和 $t(\cdot)$ 是任意神经网络（不需要可逆）。这个变换的 Jacobian 是三角矩阵，行列式 = 对角元素之积，$O(d)$ 就能算。

多层 coupling layer 堆叠，每层交换哪一半保持不动，就能构建复杂的变换。

#### 我在 SBI 中怎么用的

在我的 SBI 项目中，normalizing flow 的角色不是"生成图片"，而是**近似后验分布**。具体来说，我训练一个**条件** normalizing flow $q_\phi(\theta | \mathbf{x})$——给定观测数据 $\mathbf{x}$，直接输出参数 $\theta$ 的后验分布。

这叫 **Neural Posterior Estimation (NPE)**。训练目标：

$$\max_\phi \mathbb{E}_{(\theta, \mathbf{x}) \sim p(\theta)p(\mathbf{x}|\theta)}\!\left[\log q_\phi(\theta | \mathbf{x})\right]$$

也就是在模拟器产生的 $(\theta, \mathbf{x})$ 数据对上最大化条件对数似然。训练完之后，给任意新观测 $\mathbf{x}_{\text{obs}}$，直接从 $q_\phi(\theta | \mathbf{x}_{\text{obs}})$ 采样就行——**一次前向传播，毫秒级出结果**。

这种"训练一次、推断无限次"的模式叫 **amortized inference**。它的问题我后面会讲。

#### Flow 的局限

1. **可逆性约束限制表达力**：高维图像（如 256×256×3 = 196608 维）上，flow 的生成质量不如 diffusion model
2. **Amortization gap**：训练分布外（OOD）的观测会导致推断质量急剧下降——这是我做 SBI 时发现的核心问题

---

### 1.2 Diffusion Model：加噪再去噪

#### 核心思想：毁掉数据，再学会修复

Diffusion model 完全放弃了"学一个可逆变换"的思路。它的策略分两步：

1. **正向过程（Forward Process）**：人为设计一个逐步加噪的过程，把任何数据变成纯高斯噪声
2. **反向过程（Reverse Process）**：训练一个神经网络，学会在每个噪声水平下去掉一小步噪声

生成时，从纯噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，反复去噪，最终得到一个干净样本 $\mathbf{x}_0$。

#### 正向过程的数学

DDPM（Ho et al. 2020）定义正向过程为一系列高斯加噪步骤：

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I})$$

其中 $\beta_t \in (0, 1)$ 是预先设定的 **noise schedule**（通常从小到大，如 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$）。

每步做的事情就是：**把上一步的结果缩小一点（乘 $\sqrt{1-\beta_t}$），再加一点噪声（方差 $\beta_t$）**。

但我们不需要一步步加噪。通过递推，可以直接从 $\mathbf{x}_0$ 跳到任意 $\mathbf{x}_t$：

定义 $\alpha_t = 1 - \beta_t$ 和 $\bar\alpha_t = \prod_{i=1}^t \alpha_i$，则：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\, (1-\bar\alpha_t)\mathbf{I})$$

等价地写成 **reparameterization** 形式：

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

这就是 DMPS 论文中的统一记法 $\mathbf{x}_t = a_t \mathbf{x}_0 + b_t \boldsymbol{\epsilon}$，其中 $a_t = \sqrt{\bar\alpha_t}$，$b_t = \sqrt{1-\bar\alpha_t}$。

**推导一下为什么能跳步**。第一步：$\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_1$。第二步代入：

$$\mathbf{x}_2 = \sqrt{\alpha_2}\mathbf{x}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2 = \sqrt{\alpha_2\alpha_1}\mathbf{x}_0 + \sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2$$

后两项是两个独立高斯的和，方差相加：$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2 = 1 - \bar\alpha_2$。

所以 $\mathbf{x}_2 = \sqrt{\bar\alpha_2}\mathbf{x}_0 + \sqrt{1-\bar\alpha_2}\boldsymbol{\epsilon}$，归纳法得到一般公式。

**当 $t \to T$ 时**，$\bar\alpha_T \to 0$，$\mathbf{x}_T \approx \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：数据信息完全被噪声淹没。

#### Score Function：概率的"上山方向"

在讲反向过程之前，需要引入一个关键概念：**score function**。

对于概率密度 $p(\mathbf{x})$，其 score function 定义为：

$$\text{score}(\mathbf{x}) := \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

这是一个和 $\mathbf{x}$ 同维度的**向量**，指向概率密度增长最快的方向。

**直觉**：想象所有自然图像在高维空间形成一个"概率山丘"。人脸聚集区概率高（山顶），随机噪声概率低（山脚）。Score 就是在每个点指向"上山方向"的箭头。**沿着 score 走 = 去噪 = 让图像变得更像自然图像**。

一个关键性质：score 不需要知道归一化常数。因为 $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log \frac{p_{\text{unnorm}}(\mathbf{x})}{Z} = \nabla_{\mathbf{x}} \log p_{\text{unnorm}}(\mathbf{x})$（$Z$ 是常数，求导后消失）。

#### $s_\theta(\mathbf{x}_t, t)$：噪声预测器，不是 score

论文里的 $s_\theta(\mathbf{x}_t, t)$ 是一个神经网络（通常是 U-Net），训练目标是：

$$\min_\theta \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|s_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon}\|^2\right]$$

其中 $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$。

**$s_\theta$ 预测的是噪声 $\boldsymbol{\epsilon}$，不是 score。** 但两者有简单的数学关系。对转移概率 $p(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$ 取 score：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t} = -\frac{\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}}{1-\bar\alpha_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar\alpha_t}}$$

所以：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \approx -\frac{s_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar\alpha_t}} = -\frac{s_\theta(\mathbf{x}_t, t)}{b_t}$$

**知道噪声 $\boldsymbol{\epsilon}$ = 知道 score**，只差一个确定性系数 $-1/b_t$。

#### 反向过程：采样方程

这是我读论文时最大的困惑。正向过程 $\mathbf{x}_t = a_t\mathbf{x}_0 + b_t\boldsymbol{\epsilon}$ 是"加噪"，那"去噪"的方程长什么样？为什么不是简单地反解？

**为什么不能直接反解？** 因为我们不知道真正的 $\boldsymbol{\epsilon}$！我们只有当前的 $\mathbf{x}_t$，不知道它是从哪个 $\mathbf{x}_0$ 加噪来的。

Ho et al. (2020) 的做法：用 Bayes 法则推导 $p(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$（如果我们知道 $\mathbf{x}_0$ 的话），发现它也是高斯分布。然后用 $s_\theta$ 估计 $\mathbf{x}_0$（因为 $\mathbf{x}_0 = (\mathbf{x}_t - b_t \boldsymbol{\epsilon})/a_t \approx (\mathbf{x}_t - b_t \cdot s_\theta)/a_t$），代入得到采样方程：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} s_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**逐项解读**：

- $\frac{1}{\sqrt{\alpha_t}}(\cdots)$：**放大信号**。正向过程每步把信号乘以 $\sqrt{\alpha_t} < 1$（缩小），反向要除回来
- $-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} s_\theta(\mathbf{x}_t, t)$：**减去估计的噪声**。$s_\theta$ 预测了 $\mathbf{x}_t$ 中的噪声成分，乘以适当系数后减掉
- $\sigma_t \mathbf{z}_t$：**加回一小点新噪声**。因为 $\mathbf{x}_{t-1}$ 仍然应该是"时刻 $t-1$ 对应的带噪版本"，完全去干净就不对了。这也保证了采样的随机性/多样性

**用 score 的视角重写**。把 $s_\theta = -b_t \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$ 代入：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t + (1-\alpha_t)\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)\right) + \sigma_t \mathbf{z}_t$$

意义非常清楚：**每一步 = 当前位置 + 沿 score 方向走一小步（去噪） + 一点随机扰动**。反复迭代，从纯噪声走到数据分布。

#### 正向 vs 反向：一张表格终结困惑

| | 正向（加噪） | 反向（采样/去噪） |
|---|---|---|
| **方向** | $\mathbf{x}_0 \to \mathbf{x}_T$ | $\mathbf{x}_T \to \mathbf{x}_0$ |
| **公式** | $\mathbf{x}_t = a_t\mathbf{x}_0 + b_t\boldsymbol{\epsilon}$ | $\mathbf{x}_{t-1} = f(\mathbf{x}_t, s_\theta) + \sigma_t\mathbf{z}_t$ |
| **已知** | $\mathbf{x}_0$（干净数据） | $\mathbf{x}_t$（当前状态） |
| **未知** | $\boldsymbol{\epsilon}$（随机抽取） | $\boldsymbol{\epsilon}$（用 $s_\theta$ 估计） |
| **用途** | 训练时造带噪数据 | 推理时生成样本 |
| **步数** | 一步到位（可以直接跳到任意 $t$） | 必须一步步走（$T \to T{-}1 \to \cdots \to 0$） |
| **需要网络？** | 不需要 | 需要（$s_\theta$ 在每步提供去噪方向） |

---

### 1.3 Flow Matching：统一 Flow 和 Diffusion

#### 动机：Diffusion 太慢了

DDPM 需要 $T = 1000$ 步采样，每步调用一次神经网络。在 V100 上生成一张 256×256 图片要几十秒。能不能少走几步？

2022-2023 年，Flow Matching（Lipman et al. 2022）和 Rectified Flow（Liu et al. 2022）同时提出了一个优雅的解决方案。

#### 核心思想：学习速度场

Flow matching 把正向过程重新理解为一条**从数据到噪声的连续路径**：

$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \beta_t \boldsymbol{\epsilon}, \quad t \in [0, 1]$$

其中 $(\alpha_t, \beta_t)$ 是时间相关的系数。不同的选择给出不同的模型：

| 模型 | $\alpha_t$ | $\beta_t$ | 路径形状 |
|---|---|---|---|
| DDPM | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | 弯曲的（非线性 schedule） |
| **Rectified Flow** | $1-t$ | $t$ | **直线**（线性插值） |
| VP-SDE | $e^{-\frac{1}{2}\int_0^t \beta(s)ds}$ | $\sqrt{1-e^{-\int_0^t \beta(s)ds}}$ | 弯曲的 |

Rectified Flow 选择最简单的直线路径：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$。

对这条路径求时间导数：

$$\frac{d\mathbf{x}_t}{dt} = -\mathbf{x}_0 + \boldsymbol{\epsilon} = \boldsymbol{\epsilon} - \mathbf{x}_0$$

这就是**真实速度场**：从数据 $\mathbf{x}_0$ 指向噪声 $\boldsymbol{\epsilon}$ 的方向，恒定不变（因为路径是直线）。

Flow matching 训练一个网络 $\mathbf{v}_\theta(\mathbf{x}_t, t)$ 来预测这个速度场：

$$\min_\theta \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\mathbf{v}_\theta(\mathbf{x}_t, t) - (\boldsymbol{\epsilon} - \mathbf{x}_0)\|^2\right]$$

训练 loss 就是简单的 MSE——和 DDPM 的噪声预测 loss 一样简单。

#### 采样：ODE 求解器

有了速度场，采样就是**反向积分 ODE**（从 $t=1$ 积分到 $t=0$）：

$$\mathbf{x}_{t-\Delta t} = \mathbf{x}_t - \mathbf{v}_\theta(\mathbf{x}_t, t) \cdot \Delta_t$$

这是最简单的 Euler 法。因为路径是直线，Euler 法的误差很小，**50 步就够了**（DDPM 需要 1000 步）。

#### 速度场和 Score 的关系

Flow matching 的速度场 $\mathbf{v}_\theta$ 和 diffusion 的 score $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$ 不是独立的概念。Ma et al. (2024, SiT) 给出了精确关系：

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) = \frac{1}{\beta_t} \cdot \frac{\alpha_t \mathbf{v}_\theta(\mathbf{x}_t, t) - \dot\alpha_t \mathbf{x}_t}{\dot\alpha_t \beta_t - \alpha_t \dot\beta_t}$$

对 Rectified Flow（$\alpha_t = 1{-}t$, $\beta_t = t$, $\dot\alpha_t = -1$, $\dot\beta_t = 1$）：

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) = \frac{1}{t} \cdot \frac{(1-t)\mathbf{v}_\theta + \mathbf{x}_t}{-t - (1-t)} = \frac{(1-t)\mathbf{v}_\theta + \mathbf{x}_t}{-t}$$

这意味着：**知道速度场 = 知道 score**。Flow matching 和 diffusion model 是同一枚硬币的两面。

#### 为什么 Flow Matching 更好？

| 维度 | DDPM | Flow Matching |
|---|---|---|
| 采样步数 | ~1000 | ~50 |
| 训练 loss | $\|s_\theta - \boldsymbol{\epsilon}\|^2$（预测噪声） | $\|\mathbf{v}_\theta - (\boldsymbol{\epsilon}-\mathbf{x}_0)\|^2$（预测速度） |
| 路径形状 | 弯曲（需要精心设计 schedule） | 直线（最优传输） |
| 理论框架 | SDE/score matching | ODE/optimal transport |
| 采样方式 | SDE（有随机噪声项） | ODE（确定性，可加噪声变 SDE） |
| 代表作 | DDPM, ADM | Rectified Flow, SiT, Stable Diffusion 3 |

---

### 1.4 三代方法在 DMPS 中的角色

| 方法 | 在 DMPS 故事中的角色 |
|---|---|
| **Normalizing Flow** | 我做 SBI 用的工具，NPE-DMPS Hybrid 的前半段（amortized 初始化） |
| **Diffusion (DDPM)** | DMPS Algorithm 1（1000步，67s） |
| **Flow Matching** | DMPS Algorithm 2（50步，4.45s），也是 FIG (ICLR 2025) 的基础 |

DMPS 论文的优雅之处在于：**它的核心 trick（闭式 likelihood score）对 DDPM 和 flow matching 都适用**，只是嵌入采样方程的方式稍有不同。

---

## 第二章：逆问题与后验采样

### 2.1 问题设定

很多科学和工程问题可以写成：

$$\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$$

- $\mathbf{x}_0 \in \mathbb{R}^N$：未知的目标信号（高分辨率图像）
- $\mathbf{A} \in \mathbb{R}^{M \times N}$：已知的退化算子（下采样、模糊核、灰度化）
- $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma_y^2\mathbf{I})$：观测噪声
- $\mathbf{y} \in \mathbb{R}^M$：我们观测到的退化信号

| 任务 | $\mathbf{A}$ 的含义 |
|---|---|
| 超分辨率 (4x) | 双三次下采样（$M = N/16$） |
| 去模糊 | 卷积模糊核（$M = N$） |
| 上色 | RGB 取均值（$M = N/3$） |
| 去噪 | 单位阵 $\mathbf{I}$（$M = N$） |

**目标**：从后验分布 $p(\mathbf{x}_0 | \mathbf{y})$ 中采样。

### 2.2 贝叶斯视角

Bayes 法则：

$$p(\mathbf{x}_0 | \mathbf{y}) = \frac{p(\mathbf{x}_0)\, p(\mathbf{y}|\mathbf{x}_0)}{p(\mathbf{y})}$$

- **先验** $p(\mathbf{x}_0)$：扩散模型隐式学到的自然图像分布
- **似然** $p(\mathbf{y}|\mathbf{x}_0) = \mathcal{N}(\mathbf{y}; \mathbf{A}\mathbf{x}_0, \sigma_y^2\mathbf{I})$：直接由观测方程给出

问题是我们没有 $p(\mathbf{x}_0)$ 的解析形式——它藏在扩散模型的参数里。

### 2.3 Posterior Score 分解

关键等式（Bayes 法则在 score 空间的版本）：

$$\underbrace{\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|\mathbf{y})}_{\text{posterior score}} = \underbrace{\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)}_{\text{prior score}} + \underbrace{\nabla_{\mathbf{x}_t}\log p(\mathbf{y}|\mathbf{x}_t)}_{\text{likelihood score}}$$

推导很简单：$\log p(\mathbf{x}_t|\mathbf{y}) = \log p(\mathbf{x}_t) + \log p(\mathbf{y}|\mathbf{x}_t) - \log p(\mathbf{y})$，对 $\mathbf{x}_t$ 求导，$\log p(\mathbf{y})$ 不含 $\mathbf{x}_t$ 所以消失。

这告诉我们：**要从 $p(\mathbf{x}_0|\mathbf{y})$ 采样，只需在原始采样方程中把 prior score 换成 posterior score**。

- **Prior score** $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$：扩散模型已经学会了（$= -s_\theta/b_t$ 或从 $\mathbf{v}_\theta$ 导出）
- **Likelihood score** $\nabla_{\mathbf{x}_t}\log p(\mathbf{y}|\mathbf{x}_t)$：**这是全部难点所在**

### 2.4 Likelihood Score 为什么难算？

注意 $p(\mathbf{y}|\mathbf{x}_t)$ **不是** $p(\mathbf{y}|\mathbf{x}_0)$。$\mathbf{x}_t$ 是加了噪声的版本，要从 $\mathbf{x}_t$ 算 $p(\mathbf{y}|\mathbf{x}_t)$ 需要对所有可能的 $\mathbf{x}_0$ 积分：

$$p(\mathbf{y}|\mathbf{x}_t) = \int p(\mathbf{y}|\mathbf{x}_0)\, p(\mathbf{x}_0|\mathbf{x}_t)\, d\mathbf{x}_0$$

这个积分不可解——因为 $p(\mathbf{x}_0|\mathbf{x}_t)$ 本身就是一个复杂的后验分布。

---

## 第三章：DPS 和 DMPS——两种近似策略

### 3.1 DPS：用 Tweedie 估计 + 反向传播

DPS（Chung et al. 2022）的策略：

**Step 1**：用 Tweedie 公式从 $\mathbf{x}_t$ 估计 $\mathbf{x}_0$：

$$\hat{\mathbf{x}}_0(\mathbf{x}_t) = \frac{1}{a_t}\!\left(\mathbf{x}_t + b_t^2 \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)\right)$$

这个 $\hat{\mathbf{x}}_0$ 是后验均值 $\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]$，需要用神经网络输出计算。

**Step 2**：把不可解的 $p(\mathbf{y}|\mathbf{x}_t)$ 近似为 $p(\mathbf{y}|\hat{\mathbf{x}}_0)$：

$$p(\mathbf{y}|\mathbf{x}_t) \approx \mathcal{N}(\mathbf{y}; \mathbf{A}\hat{\mathbf{x}}_0(\mathbf{x}_t),\, \sigma_y^2\mathbf{I})$$

**Step 3**：对 $\mathbf{x}_t$ 求梯度。问题来了——$\hat{\mathbf{x}}_0(\mathbf{x}_t)$ 依赖于 $s_\theta(\mathbf{x}_t, t)$，而 $s_\theta$ 是一个 U-Net！所以求 $\nabla_{\mathbf{x}_t}\log p(\mathbf{y}|\mathbf{x}_t)$ 需要**反向传播穿过整个去噪网络**。

**代价**：每个采样步 = 一次前向 + 一次反向传播 = 双倍计算量 + 存储中间激活值。DDPM 版本（1000步）需要 194 秒。

### 3.2 DMPS：Uninformative Prior + 闭式解

DMPS 完全跳过了 Tweedie 估计和反向传播。

**Assumption 1（Uninformative Prior）**：假设 $p(\mathbf{x}_0)$ 关于 $\mathbf{x}_t$ 是 flat 的，即 $p(\mathbf{x}_0|\mathbf{x}_t) \propto p(\mathbf{x}_t|\mathbf{x}_0)$。

在这个假设下，$p(\mathbf{x}_0|\mathbf{x}_t)$ 变成了一个我们能精确写出的高斯分布。完整推导如下：

由 $p(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(a_t\mathbf{x}_0, b_t^2\mathbf{I})$，写出 log-density 关于 $\mathbf{x}_0$ 的部分：

$$\log p(\mathbf{x}_t|\mathbf{x}_0) = -\frac{1}{2b_t^2}\|\mathbf{x}_t - a_t\mathbf{x}_0\|^2 + C$$

展开，提取 $\mathbf{x}_0$ 的各阶项，配方：

$$= -\frac{a_t^2}{2b_t^2}\left\|\mathbf{x}_0 - \frac{\mathbf{x}_t}{a_t}\right\|^2 + C'$$

因此：

$$p(\mathbf{x}_0|\mathbf{x}_t) \approx \mathcal{N}\!\left(\frac{\mathbf{x}_t}{a_t},\, \frac{b_t^2}{a_t^2}\mathbf{I}\right)$$

有了这个高斯近似，$\mathbf{x}_0$ 可以写成 $\mathbf{x}_0 = \frac{\mathbf{x}_t}{a_t} + \frac{b_t}{a_t}\mathbf{w}$，$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。代入 $\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$：

$$\mathbf{y} = \frac{\mathbf{A}\mathbf{x}_t}{a_t} + \frac{b_t}{a_t}\mathbf{A}\mathbf{w} + \mathbf{n}$$

两个独立高斯的和（$\frac{b_t}{a_t}\mathbf{A}\mathbf{w}$ 和 $\mathbf{n}$）的方差相加，得到 **pseudo-likelihood**：

$$\tilde{p}(\mathbf{y}|\mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{y};\, \frac{\mathbf{A}\mathbf{x}_t}{a_t},\, \sigma_y^2\mathbf{I} + \frac{b_t^2}{a_t^2}\mathbf{A}\mathbf{A}^T\right)$$

对 $\mathbf{x}_t$ 求梯度（多元高斯 log-density 的标准梯度公式）：

$$\nabla_{\mathbf{x}_t}\log\tilde{p}(\mathbf{y}|\mathbf{x}_t) = \frac{1}{a_t}\mathbf{A}^T\!\left(\sigma_y^2\mathbf{I} + \frac{b_t^2}{a_t^2}\mathbf{A}\mathbf{A}^T\right)^{-1}\!\left(\mathbf{y} - \frac{\mathbf{A}\mathbf{x}_t}{a_t}\right)$$

**这个公式里的每一项——$\mathbf{A}$、$a_t$、$b_t$、$\sigma_y$、$\mathbf{x}_t$、$\mathbf{y}$——全是已知量，不涉及神经网络。** 这就是为什么 DMPS 不需要反传。

用 SVD $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ 进一步加速，矩阵求逆变成对角阵逐元素取倒数。SVD 只做一次，每步采样只需矩阵-向量乘法。

### 3.3 最终算法：在采样方程上"加两行"

**DDPM 版**：

$$\mathbf{x}_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}s_\theta(\mathbf{x}_t,t)\right) + \tilde\sigma_t\mathbf{z}_t}_{\text{原始无条件采样}} + \lambda\frac{1-\alpha_t}{\sqrt{\alpha_t}}\nabla_{\mathbf{x}_t}\log\tilde{p}(\mathbf{y}|\mathbf{x}_t)$$

**Flow 版**：

$$\mathbf{x}_{t-1} = \underbrace{\mathbf{x}_t - \mathbf{v}_\theta(\mathbf{x}_t,t)\Delta_t}_{\text{原始无条件采样}} - \lambda\frac{b_t(\dot\alpha_t b_t - \alpha_t\dot b_t)}{\alpha_t}\nabla_{\mathbf{x}_t}\log\tilde{p}(\mathbf{y}|\mathbf{x}_t)\,\Delta_t$$

结构完全一样：**原始采样方程 + $\lambda \times$ 闭式 likelihood score**。

| 方法 | DDPM (1000步) | Flow (50步) |
|---|---|---|
| DMPS | 67s | **4.45s** |
| DPS | 194s | 8.04s |
| PGDM | 182s | 6.44s |

---

## 第四章：我的 SBI 项目和 OOD 边界

### 4.1 SBI 是什么？

Simulation-Based Inference 解决的问题：**模拟器能跑，但似然函数写不出来**。

比如流行病传播：给定传染率 $\beta$ 和恢复率 $\gamma$，我可以用随机 SIR 模型模拟一次疫情，得到感染人数时间序列。但 $p(\text{数据}|\beta, \gamma)$ 的解析形式？不存在。

贝叶斯推断需要似然函数来计算后验 $p(\theta|\mathbf{x})$。没有似然怎么办？

**ABC（Approximate Bayesian Computation）**：最暴力的方法——反复从先验采 $\theta$、跑模拟、和观测比较、接近就留下。正确但极慢。

**NPE（我的方法）**：训练一个 normalizing flow $q_\phi(\theta|\mathbf{x})$ 直接逼近后验。Amortized：训练一次，之后对任何新观测毫秒出结果。

### 4.2 我的实验和发现

我在随机 SIR 模型上训练了条件 normalizing flow。一个关键的设计是 **CLT conditioning features**：不直接把原始观测 $(y, N)$ 喂进网络，而是用 $(y/N, \log(1+y), \log(1+N))$。其中 $y/N$ 是中心极限定理给出的一致估计量——这相当于把渐近统计结构注入网络，帮助泛化。

**结果**：
- 训练分布内 ($R_\infty = 100$-$500$)：KS $< 0.05$，和 ABC ground truth 几乎完全吻合
- 训练分布边缘 ($R_\infty = 700$)：仍然 OK
- 训练分布外 ($R_\infty = 900$+)：**崩溃**——后验偏移、方差错误

**这就是 amortization gap 的硬边界**。flow 只在训练时见过的数据范围内有效。超出这个范围，garbage in, garbage out。

这个发现直接把我引向了 DMPS——因为 per-observation 方法没有训练分布的概念，自然没有 OOD 问题。

---

## 第五章：NPE-DMPS Hybrid

### 5.1 动机

两种方法的优劣刚好互补：

| | NPE (amortized) | DMPS (per-observation) |
|---|---|---|
| 速度 | 毫秒 | 秒-分钟 |
| 训练分布内精度 | 好 | 好 |
| OOD | 崩溃 | 不受影响 |
| 需要预训练？ | 需要条件 flow | 需要无条件扩散模型 |

能不能取两者之长？

### 5.2 方法

1. 从训练好的 NPE flow 采样 $\theta_{\text{NPE}}$（毫秒级，给出一个粗略样本）
2. 给 $\theta_{\text{NPE}}$ 加噪到中间时刻 $t_{\text{start}} = \rho \cdot T$：

$$\theta_{t_{\text{start}}} = a_{t_{\text{start}}} \cdot \theta_{\text{NPE}} + b_{t_{\text{start}}} \cdot \boldsymbol{\epsilon}$$

3. 从 $t_{\text{start}}$ 开始跑截断的 DMPS reverse process 到 $t = 0$

**直觉**：如果 NPE 的样本已经接近真后验，加噪后的 $\theta_{t_{\text{start}}}$ 也接近扩散过程在 $t_{\text{start}}$ 自然产生的结果。我们跳过了前 $(1-\rho)$ 的步骤。

$\rho$ 控制"信任 NPE 多少"：$\rho = 0$ 完全信任 NPE（不跑 DMPS），$\rho = 1$ 完全不信任（从纯噪声开始）。

### 5.3 结果

在 5 维参数、8 维观测的高斯线性逆问题上：

| 方法 | C2ST $\downarrow$ | 速度 |
|---|---|---|
| 纯 NPE | 0.876 | 0.13s |
| 纯 DMPS | 0.697 | 1.90s |
| Hybrid ($\rho$=0.2, $\lambda$=1.0) | **0.567** | 0.73s |
| Hybrid + SAIP | 0.674 | 0.73s |

C2ST（Classifier Two-Sample Test）越接近 0.5 越好。0.5 = 和真实后验无法区分。

**最佳配置在 $\rho = 0.2$**——只需要 20% 的 DMPS 步数来精修 NPE 的初始化，比纯 DMPS 快 2.6 倍还更准。

### 5.4 一个意外发现：低维 per-observation 不 work

我还测试了 per-observation flow posterior sampling（Meng 的 FIG 思路）在 1D SIR 问题上的表现。

**结果：即使用精确似然做 guidance，per-observation 效果不如 amortized。**

我的分析——三个原因：

1. **先验太弱**：1D 上的无条件 flow 只能表示 Gamma(2, 1/3)。图像上的先验是在百万张图片上训练的，表达力完全不同
2. **$\lambda$ 极度敏感**：最优 guidance 权重是 $\lambda = 0.001$，极小。稍大就发散。说明低维空间里似然梯度的 scale 容易压倒先验的速度场
3. **Euler 积分误差**：100 步 ODE 可能不够精确

**这个负结果引出了一个研究假设**：per-observation 后验采样的效果可能根本性地依赖于先验模型的质量和维度。如果是，那 hybrid 方法才是低维科学问题的正确范式。

---

## 第六章：面试准备与反思

### 技术之外的准备

面试前一小时我慌了。但冷静下来后发现，我其实有三张王牌：

1. **我已经做了实验** — 不是空谈，有真实数据
2. **我有他可能没想过的 insight** — 低维 per-observation 不 work 的负结果
3. **我是两个社区的桥梁** — SBI 社区和 inverse problem 社区几乎零交叉引用

### 定位

我去不是当他的 diffusion 学生。我是带着 SBI 经验去的，要和他的 diffusion 方法合作打一个两个社区都没打过的靶子。

### 暑研方向

在 flow matching 的统一框架下，系统对比三种后验估计方法：

1. **Amortized conditional flow matching**（我已有代码）
2. **Per-observation FIG 式 guidance**（Meng 的方法带到 SBI）
3. **Hybrid**：amortized 初始化 + FIG 修正

核心假设：per-observation 方法的优势随问题维度增加而增强。在 1D，amortized 赢。在 256$\times$256 图像空间，per-observation 赢。**交叉点在哪里？** 这是一个没人回答过的问题。

---

## 写在最后

这次面试准备让我真正理解了一件事：**读论文不是目的，把两篇论文的方法接起来才是研究**。

DMPS 论文里的每一个公式都不难——forward process 是高斯加噪，uninformative prior 是配方，likelihood score 是高斯梯度，SVD 是线性代数。难的是看到 NPE 和 DMPS 之间那个没人走过的桥，然后真的动手搭起来。

> **一句话总结**：Amortized 方法快但有 OOD 边界，per-observation 方法准但慢且低维不 work。把它们接起来——让 flow 提供起点、让 diffusion guidance 精修——可能是科学逆问题上的正确范式。
