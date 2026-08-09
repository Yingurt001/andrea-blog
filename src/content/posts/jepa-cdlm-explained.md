---
title: JEPA-CDLM 到底在做什么？从 JEPA 分支到完整连续潜空间语言模型
published: 2026-08-09
description: 从 T5 contextual feature、确定性 TextAE、EMA teacher、JEPA predictor 和 Flow Matching 开始，逐步解释 JEPA-CDLM 的训练目标、梯度路径与推理过程。
image: /images/jepa-cdlm-explained/cover.webp
tags:
  - JEPA
  - Continuous Latent
  - Flow Matching
  - Diffusion Language Model
  - NLP
category: 科研
draft: false
---

# JEPA-CDLM 到底在做什么？

上一篇[连续潜空间语言模型横向对比](/posts/continuous-latent-language-models-comparison/)写到我们自己的 JEPA-CDLM 时，我用了很多已经压缩过的信息：online compressor、EMA teacher、$t=1$ predictor、Flow endpoint、stop-gradient。每个词单独看都能查到解释，放进同一个模型以后却很难知道它们之间是什么关系。

这篇文章从头介绍 JEPA-CDLM。它不默认读者已经理解 JEPA，也不默认读者知道 Flow Matching 如何生成文本。我们先回答一个基本问题：这个模型最终要学会什么？然后再分别解释 T5、TextAE、JEPA branch、Flow branch 和 Decoder。

如果你还没有接触过 Encoder、Decoder 和 latent space，可以先读[《Autoencoder 到底在做什么？》](/posts/autoencoder-explained/)。这篇会继续使用这些概念，但会把模型中的 tensor shape 和 loss 写清楚。

## 我们要解决的是什么问题？

语言是离散的。一段文本经过 tokenizer 后会变成 token ids，例如：

```text
[71, 19, 842, 6, 1, ...]
```

Flow Matching 更适合处理连续变量，因此我们先把 token 转成连续 latent，再学习如何从 Gaussian noise 生成这些 latent：

```text
token
  → T5 contextual feature
  → TextAE compressed latent
  → Flow 从噪声生成 latent
  → TextAE Decoder
  → token
```

问题是，Autoencoder 只要求 latent 能够重构原文，Flow 却要求 latent distribution 容易从噪声中学习。两种要求不完全相同。

一个高容量 Autoencoder 可以给每个训练样本分配非常精确的 latent。Decoder 能读，reconstruction CE 也很低。但是只要 Flow prediction 偏离 clean latent 一点，Decoder 的输出就可能快速变差。

JEPA-CDLM 因此增加了一条 predictive representation learning 路径：先破坏 target feature，再要求模型在 latent space 中恢复 clean target representation。我们希望压缩器学到的表示既能被 Decoder 读取，也能从不完整或带噪的信息中预测。

## 先看完整架构

![JEPA-CDLM 完整训练架构](/images/jepa-cdlm-explained/architecture-overview.svg)

训练时有三条主要路径：

1. JEPA/readout path：从受损 target representation 预测 clean latent，然后重构 feature 和 token。
2. EMA teacher path：读取 clean target，提供变化较慢的监督目标 $z^*$。
3. Flow Matching path：从 Gaussian noise 出发，学习生成同一个 $z^*$。

当前默认 loss 可以写成：

$$
\mathcal L
=\lambda_{\mathrm{CE}}\mathcal L_{\mathrm{CE}}
+\lambda_{\mathrm{feat}}\mathcal L_{\mathrm{feat}}
+\lambda_{\mathrm{JEPA}}\mathcal L_{\mathrm{JEPA}}
+\lambda_{\mathrm{Flow}}\mathcal L_{\mathrm{Flow}}
+\lambda_{\mathrm{SigReg}}\mathcal L_{\mathrm{SigReg}}.
$$

基础配置中前四项的权重都是 1，SigReg 的权重是 0.1。不同实验配置可以改变这些值，因此它们不是模型定义中不可修改的常数。

## 第一部分：token 如何变成 compressed latent？

### 冻结的 T5-small 做什么？

假设 LM1B 输入画布有 128 个 token。T5-small 为每个位置输出一个 512 维 contextual feature：

$$
[B,128]
\xrightarrow{\text{frozen T5-small}}
[B,128,512].
$$

contextual feature 和普通 token embedding 不同。同一个 token 出现在不同上下文中，经过 T5 self-attention 后会得到不同的 512 维表示。

T5 参数被冻结。训练时不会修改 T5，只训练后面的 compressor、DiT、Decoder 和 unembed。这样可以让 TextAE 的输入坐标保持稳定，但也意味着 T5 feature 本身不会根据 Flow objective 继续适应。

### 当前 TextVAE 为什么实际上是 TextAE？

标准 VAE 会输出 posterior 的均值和方差：

$$
\mu(x),\qquad \log\sigma^2(x),
$$

然后通过重参数化进行采样，并加入 KL loss。当前代码中的 TextVAE 没有这些操作：

- 没有 $\mu$ 和 $\log\sigma$；
- 没有 posterior sampling；
- 没有 KL loss；
- 同一个输入确定性地得到同一个 latent。

因此它更准确的名字是 deterministic TextAE。

### 4× length compression 是怎么做的？

TextAE 将相邻四个 T5 features 拼接：

$$
[h_1;h_2;h_3;h_4]\in\mathbb R^{4\times512}
=\mathbb R^{2048}.
$$

然后执行：

```text
4×512 feature
→ Linear(2048, 768)
→ prefix-bidirectional Transformer
→ Linear(768, Dz)
```

在无条件 LM1B 中：

$$
[B,128,512]\rightarrow[B,32,D_z].
$$

TextAE 有两个维护中的 profile：

| profile | Encoder / Decoder layers | Transformer width | $D_z$ |
| --- | ---: | ---: | ---: |
| `tiny` | 6 / 6 | 768 | 512 |
| `small` | 12 / 12 | 768 | 768 |

这里的 `small` 是 TextAE profile，不是前面的 T5-small。

对于条件生成，condition 和 target 会分别 patch：

$$
S_{\mathrm{cond}}=\left\lceil\frac{L_{\mathrm{cond}}}{4}\right\rceil,
\qquad
S_{\mathrm{target}}=\left\lceil\frac{L_{\mathrm{target}}}{4}\right\rceil.
$$

这样可以避免一个 patch 同时包含 condition token 和 target token。condition query 只读取 condition，target query 可以读取 condition 和 target，因此 condition representation 在推理时不需要真实 target。

## 第二部分：JEPA 是什么？

JEPA 是 Joint-Embedding Predictive Architecture。它的基本目标不是直接恢复原始输入，而是从一个 view 的 representation 预测另一个 view 的 representation。

以 I-JEPA 的标准形式为例，context encoder 读取可见区域，target encoder 读取目标区域，predictor 根据 context representation 预测 target representation。目标在 embedding space 中定义，而不是直接要求模型逐像素生成图像。[I-JEPA 原论文](https://arxiv.org/abs/2301.08243)

online/target 双网络和 EMA update 也与 BYOL 一类方法有关。BYOL 使用 online network 预测 target network 的表示，并通过 online 参数的 moving average 更新 target network。[BYOL 原论文](https://arxiv.org/abs/2006.07733)

我们的实现沿用了以下结构：

```text
受损 view → online encoder → predictor → predicted representation

干净 view → EMA target encoder ─────────→ target representation
```

然后比较两者：

$$
\mathcal L_{\mathrm{JEPA}}
=\left\|z_{\mathrm{pred}}-\operatorname{sg}(z^*)\right\|_2^2.
$$

$\operatorname{sg}$ 表示 stop-gradient。

## stop-gradient 是什么意思？

假设 loss 是：

$$
\left\|z_{\mathrm{pred}}-z^*\right\|^2.
$$

如果两边都接收 gradient，online encoder 和 teacher encoder 会同时追着对方变化。stop-gradient 将 $z^*$ 当作当前 step 的固定监督值：

```python
z_target = z_teacher.detach()
loss = mse(z_pred, z_target)
```

因此 gradient 会更新：

- online compressor；
- predictor，也就是共享 DiT 的 terminal decode branch。

gradient 不会更新 EMA compressor。EMA compressor 通过另一条规则更新：

$$
\theta_{\mathrm{teacher}}
\leftarrow
\tau\theta_{\mathrm{teacher}}
+(1-\tau)\theta_{\mathrm{online}}.
$$

如果 $\tau=0.996$，一次更新中 teacher 保留 99.6% 的旧参数，加入 0.4% 的 online 参数。当前默认 schedule 会把 $\tau$ 从 0.996 逐渐增加到 1.0；到 1.0 后，teacher compressor 不再变化。

## 第三部分：我们的 JEPA 分支具体怎么走？

![JEPA 分支的数据流和 gradient](/images/jepa-cdlm-explained/jepa-branch.svg)

### 第一步：构造 online corrupted view

同一段 `[condition | target]` 文本先经过冻结 T5，得到 clean feature $h$。

训练时只破坏 target positions。condition 保持 clean，因为条件生成在推理时必须使用真实、未受损的 condition。

默认 corruption 是给 target T5 feature 加 Gaussian noise：

$$
\tilde h_{\mathrm{target}}
=h_{\mathrm{target}}+\sigma\epsilon.
$$

代码还支持：

- `latent_to_0`：将一部分 target feature 置零；
- `latent_mask`：让一部分 target token 不进入 compressor attention；
- 其他 noise-mix ablation。

Online TextAE 读取 `[clean condition | corrupted target]`，得到：

$$
z_{\mathrm{corrupt}}=E_{\mathrm{online}}(\tilde h).
$$

### 第二步：构造 clean teacher target

EMA compressor 读取未被破坏的 clean feature：

$$
z^*=E_{\mathrm{EMA}}(h).
$$

这里 EMA teacher 只复制 compressor。它不是另一套 T5，也没有单独复制 TextAE Decoder。T5 本来就是冻结的，Decoder 仍然是 online、可训练并且共享的。

$z^*$ 随后 detach，作为 JEPA 和 Flow 的 clean target endpoint。

### 第三步：DiT 在 $t=1$ 模式预测 clean latent

当前模型使用同一个 12-block、768-width DiT trunk 完成两种任务：

- random-$t$ Flow prediction；
- $t=1$ terminal JEPA prediction。

两个任务共享 Transformer trunk，但使用不同的 output head：

```text
shared DiT trunk
├─ final_layer          → Flow head
└─ decode_final_layer   → JEPA / readout head
```

JEPA 分支将 $z_{\mathrm{corrupt}}$ 和 clean condition slots 输入 DiT，并把 model time 设为 $t=1$：

$$
z_{\mathrm{JEPA}}
=P_{\theta}(z_{\mathrm{corrupt}},z_{\mathrm{cond}},t=1).
$$

这里的 $t=1$ 既表示我们采用的 Flow convention 中 clean data 所在的一端，也用于选择 terminal decode mode。输入的 $z_{\mathrm{corrupt}}$ 并不一定是 Flow trajectory 上严格的 $t=1$ state，因此代码为这个任务保留了独立的 `decode_final_layer`。

condition slots 不需要重新预测。DiT 只输出 target slots，之后再把 clean online condition 拼回去：

$$
z_{\mathrm{readout}}
=[z_{\mathrm{cond}}\mid z_{\mathrm{JEPA}}].
$$

### 第四步：在 latent、feature 和 token 三个位置计算 loss

第一个 loss 是 latent-space JEPA MSE：

$$
\mathcal L_{\mathrm{JEPA}}
=\left\|z_{\mathrm{JEPA}}-\operatorname{sg}(z^*)\right\|^2.
$$

第二个 loss 在 T5 feature space 中计算。TextAE Decoder 把 32 个 slots 展回 128 个 feature：

$$
\hat h=D_{\theta}(z_{\mathrm{readout}}),
$$

然后比较 $\hat h$ 和 clean T5 feature：

$$
\mathcal L_{\mathrm{feat}}
=\left\|\hat h-h\right\|^2.
$$

第三个 loss 是 token CE。unembed 将每个 512 维 feature 变成 vocabulary logits：

$$
\mathcal L_{\mathrm{CE}}
=-\sum_i\log p_\theta(x_i\mid z_{\mathrm{readout}}).
$$

默认配置中 CE 监督 target positions，不要求模型预测 condition token。EOS-fill 是否进入 CE 由具体实验配置决定。

## 为什么需要三个 readout loss？

它们约束的对象不同。

| Loss | 比较位置 | 它解决的问题 |
| --- | --- | --- |
| JEPA MSE | compressed latent | 预测结果是否接近 clean teacher coordinates |
| feature MSE | token-aligned T5 feature | Decoder 展开后是否恢复 contextual representation |
| token CE | vocabulary logits | 最终离散 token 是否正确 |

只使用 token CE 时，只要 Decoder 能输出正确 token，中间 latent 可以存在很多不同的解。JEPA MSE 给 predictor 一个直接、dense 的 latent target。feature MSE 则防止 unembed 单独承担全部重构任务。

反过来，只使用 JEPA MSE 也不够。latent 距离接近不保证 Decoder 一定能读出正确 token，所以模型仍然需要 feature MSE 和 CE 检查 readout quality。

## 第四部分：Flow Matching 分支做什么？

JEPA 分支看到的起点仍然来自 target 的 corrupted representation。即使 corruption 很强，它仍然包含部分真实 target 信息，因此 JEPA 本身不是完整生成模型。

Flow Matching 负责从纯 Gaussian noise 生成 target latent。

项目使用以下时间方向：

$$
t=0:\ \epsilon\sim\mathcal N(0,I),
\qquad
t=1:\ z^*.
$$

训练时随机采样 $t$，构造线性路径：

$$
z_t=(1-t)\epsilon+t z^*.
$$

对应的目标速度是：

$$
v^*=z^*-\epsilon.
$$

当前 DiT 的 public prediction type 是 `x_pred`，也就是先预测 clean endpoint，再转换成 velocity 计算默认 Flow loss。代码也提供直接在 x-domain 计算 loss 的 ablation。

Flow Matching 的基本思想是直接回归概率路径上的 vector field，推理时再用 ODE solver 从噪声积分到数据端。[Flow Matching 原论文](https://arxiv.org/abs/2210.02747)

## JEPA 和 Flow 是否在重复做同一件事？

两条分支都把输出对齐到 $z^*$，但输入分布和训练目的不同。

| 分支 | 输入 | 目标 | 主要作用 |
| --- | --- | --- | --- |
| JEPA | corrupted online target latent | clean EMA latent $z^*$ | 学习 representation recovery，并训练 terminal readout |
| Flow | Gaussian 与 $z^*$ 之间的 $z_t$ | 同一个 $z^*$ / velocity | 学习从噪声到数据的完整生成路径 |

JEPA 不能代替 Flow，因为它没有训练从纯噪声到 target distribution 的整个过程。

Flow 也不能完全代替 JEPA。Flow 可以学习如何逼近当前 clean endpoint，但默认 EMA target 是 stop-gradient。JEPA 则让 corrupted online target path、terminal predictor 和 Decoder 直接受到 clean teacher、feature 和 token 的联合约束。

因此我们的设计不是训练两个生成器，而是让两个预测问题共享同一种 target representation：

```text
corrupted representation → clean representation
Gaussian noise           → clean representation
```

## 哪些 loss 会更新哪些模块？

这是理解 JEPA-CDLM 最容易混乱的部分。

| 模块 | JEPA / CE / feature | Flow | SigReg | EMA update |
| --- | ---: | ---: | ---: | ---: |
| Frozen T5 | 否 | 否 | 否 | 否 |
| Online target compressor | 是 | 默认 EMA 模式下 target endpoint 不接收 Flow gradient | 是 | 否 |
| Online condition compressor | 是 | 条件生成时可以接收 Flow condition gradient | 是 | 否 |
| EMA compressor | 否 | 否 | 否 | 是 |
| Shared DiT trunk | 是 | 是 | 否 | 否 |
| Decode final head | 是 | 否 | 否 | 否 |
| Flow final head | 否 | 是 | 否 | 否 |
| TextAE Decoder / unembed | 是 | 否 | 否 | 否 |

表中的 target path 和 condition path 使用的是同一个 online compressor 参数，并不是两套独立模块。分开写是为了说明 gradient 从哪些 slot population 进入共享 compressor。

对于 LM1B 这类真正无条件的数据，没有 condition slots，因此 Flow condition gradient 这一项不存在。默认 `denoise_grad_yes=false` 时，Flow loss 不会通过 target endpoint 更新 online target compressor。

代码提供 `denoise_grad_yes` 等半梯度实验开关，但这些不是基础 JEPA-AE 定义，不能和默认方法混在一起介绍。

## 为什么还需要 SigReg？

如果 online compressor 和 teacher compressor 都把所有输入映射成同一个常量，JEPA MSE 也可以接近零：

$$
z_{\mathrm{JEPA}}=z^*=c.
$$

这叫 representation collapse。

实际模型还有 feature MSE 和 token CE，因此完全 collapse 会破坏重构。但项目仍然在 online latent 上加入 SigReg，直接约束 latent distribution，避免方差过低或表示集中到少数方向。

SigReg 只作用于 online compressor 输出，不作用于 detached EMA teacher。VICReg 接口也存在，但基础配置使用的是 SigReg，而不是同时启用两套 regularizer。

## 项目里为什么有两个 EMA？

当前配置中容易看到两组 EMA：

1. JEPA teacher EMA：只复制 compressor，用于产生 clean target $z^*$。
2. Model weight EMA：维护整个可训练模型的平滑权重，主要用于 evaluation 和 checkpoint。

它们不是同一件事。前者参与训练目标的构造，后者主要决定评估时使用哪组模型权重。

## 训练结束后如何生成文本？

推理时没有真实 target，所以不能构造 $z_{\mathrm{corrupt}}$，也不需要 EMA teacher。

条件生成的实际过程是：

```text
condition token
→ frozen T5
→ online TextAE compressor
→ clean condition slots

Gaussian target slots
→ ODE rollout with Flow head
→ z_ODE
→ t=1 Predictor / decode head
→ predicted target slots

[clean condition slots | predicted target slots]
→ TextAE Decoder
→ unembed
→ token
```

无条件 LM1B 没有 condition path，所有 slot 都属于 target。

## 当前训练和推理之间还有什么问题？

![JEPA-CDLM 训练与推理路径对比](/images/jepa-cdlm-explained/training-vs-inference.svg)

当前默认配置是：

```yaml
decode_from_flow_prediction: false
```

因此训练时 $t=1$ predictor 的 target input 是：

$$
z_{\mathrm{corrupt}}.
$$

推理时同一个 predictor 接收的是 ODE rollout 产生的：

$$
z_{\mathrm{ODE}}.
$$

训练直接约束了：

$$
P(z_{\mathrm{corrupt}})\rightarrow z^*,
$$

却没有直接约束：

$$
P(z_{\mathrm{ODE}})\rightarrow
\text{Decoder-readable latent}.
$$

代码已经提供 `decode_from_flow_prediction=true` 的全开/全关实验，但它会把整个 predictor readout 切到 Flow prediction。我们正在考虑的 late-Flow mixture 更细：只让一部分训练样本使用 detached late-$t$ Flow prediction，其余样本保留 corrupted 和 clean anchor。

这一方案目前仍然是待验证 hypothesis，不能写成已经完成的贡献。

## 我们现在可以声称什么？

代码中已经实现：

- frozen T5 feature surface；
- 4× TextAE length compression；
- online compressor 和 EMA compressor；
- target-only corruption；
- shared DiT trunk 与独立 Flow/decode final heads；
- JEPA MSE、feature MSE、token CE、Flow loss 和 SigReg；
- ODE rollout 后的 terminal $t=1$ readout；
- clean、corrupted、Flow prediction 和 ODE terminal 的 probe 接口。

但需要把实验边界说明清楚。目前最完整的 generation evidence 主要来自前一版 Cosmos/Perceiver compressor 谱系，不是 TextAE profile 的最终成绩单。旧实验可以支持 corruption geometry、gradient route 和 readout mismatch 的诊断，不能直接证明 TextAE 已经优于其他 compressor。

因此当前更合适的研究问题是：

> predictive representation learning 能否减少 continuous latent language model 中的 Decoder-interface mismatch，并让 Flow 生成的 latent 更稳定地被 Decoder 读取？

“首次把 JEPA 用在 latent diffusion language model”不是一个足够稳的贡献表述。JEPA、BYOL、denoising Autoencoder 和 EMA self-distillation 都提供了相邻思想。我们需要证明的是具体机制，而不是组件名称。

## 新手最容易混淆的几个问题

### JEPA predictor 是 Decoder 吗？

不是。

JEPA predictor 输入和输出都是 compressed latent。TextAE Decoder 则把 compressed latent 展开成 token-aligned T5 feature。两者串联：

```text
z_corrupt
→ JEPA predictor
→ z_JEPA
→ TextAE Decoder
→ T5 feature
→ unembed
→ token
```

### EMA teacher 是一套完整模型吗？

不是。当前 `teacher=ema` 只 deep-copy compressor。T5 冻结且共享，Decoder 和 unembed 只有在线可训练版本。

### JEPA branch 能单独生成文本吗？

不能。它训练的是从 corrupted representation 到 clean representation 的预测。真正从 Gaussian noise 生成 target latent 仍然依赖 Flow Matching。

### 为什么 JEPA loss 不直接预测 token？

因为我们希望先约束 compressed latent geometry。如果只使用 token CE，Decoder 可以吸收一部分误差，latent prediction 本身未必接近 clean endpoint。JEPA MSE 给 latent predictor 一个直接目标。

### target encoder 训练时能看到 target，会不会数据泄漏？

teacher target 本来就是由真实 target 构造的监督信号，与训练分类器时使用 label 类似。推理时不会把真实 target latent送给 Flow。需要保证的是 condition path 不能读取 target，这一点由 prefix mask 控制。

### 为什么叫 $t=1$ predictor？

本项目约定 $t=0$ 是 Gaussian noise，$t=1$ 是 clean data。terminal predictor 使用 $t=1$ time embedding 和独立 decode head，输出 Decoder 要读取的 clean target latent。

## 最后重新走一遍完整数据流

训练时：

```text
1. token → frozen T5 → clean contextual features
2. target feature 加噪 → Online TextAE → z_corrupt
3. clean feature → EMA compressor → stop-grad z*
4. z_corrupt → DiT t=1 decode head → z_JEPA
5. JEPA MSE: z_JEPA 对齐 z*
6. z_JEPA → TextAE Decoder → feature MSE + token CE
7. Gaussian 与 z* 构造 z_t → DiT Flow head → Flow loss
8. Online latent → SigReg，避免 representation collapse
```

推理时：

```text
1. condition → frozen T5 + Online TextAE → condition slots
2. target slots 从 Gaussian noise 初始化
3. Flow ODE: t=0 → t=1
4. ODE terminal → t=1 decode head
5. target slots + condition slots → TextAE Decoder
6. unembed 并行输出 token
```

JEPA 分支的作用可以压缩成一句话：

> 它用稳定的 clean teacher latent 监督受损 online latent 的恢复，同时通过 feature MSE 和 token CE 检查恢复后的 latent 是否仍然可解码；Flow 分支再学习从纯噪声生成同一种 latent。

---

## 参考资料

- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [连续潜空间语言模型横向对比](/posts/continuous-latent-language-models-comparison/)
- [Autoencoder 到底在做什么？](/posts/autoencoder-explained/)

> 注：本文以当前 `jepa_ae` + deterministic TextAE 代码路径为主。项目仍保留 Cosmos、direct patchify、teacher-free、denoise-gradient 和 STC 等实验接口，这些开关不属于本文介绍的基础配置。
