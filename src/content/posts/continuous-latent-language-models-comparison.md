---
title: 横向对比所有的连续潜空间语言模型：九个模型到底在忙什么？
published: 2026-08-09
description: 把 AURORA-LM、LTF、ELF、Cola-DLM、LDLM、Cosmos、TextLDM、FlowMapLM 与 LangFlow 塞进同一张 Encoder–Denoiser–Decoder 地图，看清每个模型究竟把难题交给了谁。
image: /images/continuous-latent-lm/cover.webp
tags:
  - Diffusion Language Model
  - Continuous Latent
  - Flow Matching
  - NLP
  - 论文阅读
category: 科研
draft: false
---

# 横向对比所有的连续潜空间语言模型

最近在学习不同的Flow matching 的 Language Model, 本来以为他们的区别在于对于Latent Space的选择，后来发现其实他们都是实际将相同的训练目标和训练压力在不同的阶段进行权衡，以一个试错的形式来推广自己的模型——Encoder, Denoiser 和 Decoder 
比如AURORA-LM、LTF、ELF、Cola-DLM、TextLDM、FlowMapLM……每篇都像在另起炉灶，但从本质来看他们所用的技巧在很多程度上是相通的，而且不难发现为什么他们会这么选择自己的模型。

> 如果你还不熟悉Autoencoder或者对 Encoder、Decoder 和 latent space，可以先读[《Autoencoder 到底在做什么？》](/posts/autoencoder-explained/)这篇文章，那篇文章用一个 Fashion-MNIST VAE 小项目把这三个词从头讲开，这篇默认你已经知道 Autoencoder 的基本数据流。

我把九个模型根据他们的训练目标和训练压力的相同点和不同点硬展开成一张表以后，发现了一些普遍的规律：**它们基本都在处理在语言FLow中会出现的相同的问题，而且解决方案本质上也是一致的，只是选择在三个不同的模块(Encoder, Denoiser 和 Decoder )上进行平摊。**

- 有作者的解决方案是：我们可以在Encoder阶段就把 latent进行压缩，那么对于Denoiser(生成器)自然就轻松了,因为只需要处理更少更平滑的信息，但是也许会给Decoder造成一些压力；
- 有作者的解决方案是：我们一点也不压缩，一个token由一个丰富的latent space进行表示，信息一丁点都别丢，我们可以把压力给 denoiser，让denoiser去学习这个高维的latent space, 加上一些训练的技巧；
- 也有作者的解决方案是：我们可以采用联合训练，Encoder、Denoiser、Decoder 不分开寻来拿，三个人一起练。
- 更有作者的解决方案是：为什么非要 Autoencoder？我直接在 one-hot 或 token embedding 上流。

所有的模型几乎都在解决同一个核心问题：语言如何进行表示和生成？
在Flow matching 领域每个模型认为困难在哪里，它又把主要的模型容量和训练技巧放在了哪里？

![连续潜空间语言模型的统一框架](/images/continuous-latent-lm/unified-framework.svg)

## Encoder–Denoiser–Decoder 三个模块可以划分成两条线

Encoder → Denoiser → Decoder 这个路径在大部分的语言流是对的，不过如果只画一条直线，很容易把训练和生成混在一起。更准确的读法是两段：

1. **先训练表示。** Encoder 把 token 变成干净 latent $z_0$，Decoder 再把它还原成 token。这里决定 latent 到底保留多少信息、空间是否平滑、Decoder 容不容错。
2. **再训练生成。** Denoiser 从噪声出发，学着生成同一种 latent。生成出的 $\hat z_0$ 最后交给 Decoder 读回文字。

中间那块 latent 是所有模块都在承担的风险

- Decoder 希望 latent 信息越全越好，最好原文每个字都别丢。
- Denoiser 希望 latent 越规整越好，最好低维、平滑、没有奇怪尖角。
- 速度又希望所有位置并行生成，连贯性却偏偏喜欢有先后顺序。

理解了这个三角矛盾，后面九个模型之间各种奇怪的设计就都不显得这么奇怪。

## 四个基本问题

### 问题一：什么是压缩？—— 压缩有长度和宽度两个维度

设一段文字有 $L$ 个 token，Encoder 输出 $K$ 个 latent，每个 latent 有 $d$ 维。总容量粗略看就是 $K\times d$。

这里有两种完全不同的“压缩”：

- **压长度**：比如把 256 个 token 变成 8 个 latent，这是 LTF 的 32 倍压缩。
- **压宽度**：仍然一个 token 对一个 latent，也就是L=K但把每个 latent 从 512 维挤到 128 维，这是 ELF 的 bottleneck。

AURORA-LM 最激进的地方反而是**不压**：最强设置取 $K=L$，宽度还保留 768 或 1024。它相当于对 Decoder 说“信息管够”，然后转头对 Denoiser 说“剩下的你想办法”。

### 问题二：谁负责让 latent 变得“好生成”？

所谓 latent 好不好，不只看能不能把原文重构回来。还得考虑从噪声生成出来、带一点误差的 latent，Decoder 还能不能读？

常见办法有三类：

- **在表示侧塑形**：KL(VAE思想)、masking、对齐冻结语言模型、扰动后恢复（我们模型的思路）。
- **在生成侧补偿**：调 noise schedule、加输入瓶颈、consistency、self-conditioning, Denoiser 压力
- **让两侧共同适应**：Encoder / Denoiser / Decoder 联合训练。

这就是我说的“把这个问题交给谁处理”。
所以在如何让latent变得好生成的问题，就衍生出了这九个甚至更多个连续语言模型，甚至未来大家还会在这个方向上继续做更多的研究，但是本质上就是在这上面做文章；

### 问题三：latent 怎么生成，和 token 怎么读出，是两回事

这是整张表最容易踩的坑。

![latent 生成顺序与 token 读出顺序是两回事](/images/continuous-latent-lm/two-orders.svg)

LTF 可以一步把全部 latent 并行生成出来，但 Decoder 仍然一个 token 一个 token 地吐；AURORA-LM 则是 latent 按块生成，但 Query-based Decoder 不接收上一时刻吐出的 token，所以读回阶段可以一次前向并行完成。

判断一个 Decoder 是否自回归，别只看 attention mask 是不是 causal，要看**推理（inference）时是否必须把上一步输出的 token 再喂回去**。

### 问题四：训练完是冻结，还是大家继续一起动？

标准两阶段做法是 Autoencoder 训完冻结，再单独训练 latent generator。好处是稳、好诊断；坏处是 Denoiser 只能迁就一块不是为它量身定做的空间。

联合训练允许 latent 跟着生成器一起改变，但也可能出现另一种问题：Encoder 找到一种 Decoder 可以解码、Denoiser 却很难学习的表示，或者整个模型退化到低多样性的输出。LDLM 和 Cola-DLM 都在直接处理这个问题。

## 九个模型，一张架构速查图

![九个连续文本生成模型架构速查](/images/continuous-latent-lm/model-atlas.svg)

下面逐个分析。对于每个模型，我主要看三个问题：它把训练压力放在哪个模块，具体用了什么方法，以及这样做有什么代价。

## 1. AURORA-LM：可生成压力在生成侧补偿

一句话概括 AURORA-LM: **把 latent 的高容量完整保留下来，再把“让它可生成”的压力集中放到 denoiser。**

它的最佳设置基本不压长度，latent 宽度也很大。这样做的好处很直白：Decoder 不缺信息，token-level fidelity 容易保住。可坏处也一样直白：高维、满长的 latent 分布不是一个好学的对象。

因此 AURORA-LM 主要在 Denoiser 上加入以下设计：

- 用低秩入口限制 denoiser 读取 noisy latent 的通道，逼它抓结构而不是记噪声细节；
- 用与 latent 维度相关的 tan-$d$ noise schedule，把训练预算重新分配；
- 用 self-trajectory consistency 缩小训练时“独立噪声档”和推理时“连续轨迹”之间的裂缝；
- 再加 self-conditioning，把自己上一步的预测喂回来。

注意，这不等于 Autoencoder 侧完全没做事。它仍然有 token embedding dropout、latent dropout 和 Query-based Decoder。更准确的说法是：**AURORA 不愿意靠压缩换一个简单空间，所以把最有辨识度的“塑性”设计放在了 denoiser 一侧。**

所以 AURORA-LM 的取舍很明确：它不通过压缩 latent 来降低生成难度，而是保留表示容量，再让 Denoiser 学习高维 latent 的分布和生成轨迹。

## 2. LTF：可生成压力在表示侧：把 256 个 token 压缩成 8 个 latent，再进行一步生成

LTF 和 AURORA-LM 的选择相反：**LTF 对 latent length 进行了很强的压缩。**256 个 token 只保留 8 个 latent，长度压缩 32 倍。

这种压缩容易产生两个 shortcut：Encoder 仍然把局部 token 信息编码进少量 latent，或者 Decoder 主要依赖前文预测下一个词，从而不使用 latent。LTF 同时限制两边：Encoder 训练时会丢弃一部分 token，Decoder 对前文的访问也会被随机切断，并且使用滑动窗口。这样会迫使压缩后的 latent 保留更多全局信息。

第二阶段使用 MeanFlow，可以一步生成全部 latent。但需要注意：**它的 latent 是并行生成的，token 仍然由自回归 Decoder 串行读回。**

因此 LTF 的方法是先把长文本压缩成少量 latent，并行生成这些 latent，再使用自回归 Decoder 将其展开成 token。它减少了 latent generation 的步数，但没有消除 token decoding 的串行过程。

## 3. ELF：可生成压力在Both：冻结 T5 表示，只压缩 latent width

ELF 直接使用冻结的 T5 Encoder 表示，不压缩序列长度，只通过线性 bottleneck 把 latent width 从 512 维降到 128 维。

它默认没有完全独立的 Decoder。同一个网络大部分时间训练 Flow Matching，小部分时间在 $t=1$ endpoint 上进行 token decoding，训练配比大致是 80% MSE 和 20% CE。换句话说，Denoiser 和 Decoder 在这里共享了大部分参数。

ELF 还比较了联合训练和标准两阶段冻结，结果的 trade-off 接近。它选择联合方案，主要是流程更省，而且能把生成 PPL 往更低处推。

ELF 对 latent representation 的改动较少。它的基本判断是：预训练语言模型的 embedding 已经包含足够的信息，**只需要进行 width compression**，再让连续 Flow 对这个空间进行适配。

## 4. Cola-DLM：可生成压力在Both：VAE 和 DiT 进行联合训练

Cola-DLM 用的是一条很规整的路线：严格因果 VAE 负责 latent，DiT + Flow Matching 负责生成，latent 不压长度，宽度从 16、64 到 128 都系统扫过。

它在表示侧使用 KL 和 BERT 式 masking。KL 用于约束 latent distribution，masking 则避免 Encoder 只复制局部 token 信息。

Cola 最该单独记住的有两点。

第一，它让 VAE 和 DiT 联合微调，不把 VAE 训完就永远冻住。第二，它发现最优 timeshift 会随着 latent 维度变化，而且不只报告经验现象，还给了理论解释。同样的噪声加在 16 维和 128 维空间里，破坏程度根本不是一回事，noise schedule 不能脱离维度来谈。

它的 latent 生成也是 block-causal：块间有先后，块内并行。质量和速度之间不再只有“全并行”与“全串行”两个档。

## 5. LDLM：可生成压力在表示侧：联合训练需要额外的稳定机制

LDLM 直接联合训练 Encoder、Diffusion Model 和 Decoder。它希望可训练 Encoder 将预训练语言模型的表示调整为一个既容易去噪、也容易解码的 latent space。

但是作者发现 naive joint training 会导致训练失败，生成多样性也会明显降低。因此他们加入了 MSE decoder loss、diffusion-to-encoder warmup、自适应时间步采样，以及 Decoder input noise。

这里比较重要的是：允许 latent 适应 Denoiser，不代表联合训练自然就会得到适合生成的 latent。如果训练日程和 loss weight 没有处理好，联合优化可能得到 reconstruction 和 training loss 都很好、生成质量却很差的解。

## 6. Cosmos：可生成压力在表示侧：重构准确不代表 latent 适合扩散

Cosmos 关心的是两个词：compressed 和 smooth。它从冻结的 BERT 类 Encoder 起步，最多把长度压到 8 倍，同时让 Autoencoder 的表示对齐冻结语言模型的内部激活。

它还做 perturb-and-recover：对 hidden state 随机置零或加噪，再要求恢复干净表示。这不是显式地罚 Jacobian 或 Lipschitz 常数，但效果上是在逼 latent 流形对小扰动更稳。

Cosmos 说明 token reconstruction 接近 100%，也不能保证 diffusion 容易学习。Autoencoder 可能把信息编码在对小扰动非常敏感的坐标中，Decoder 可以读取这种表示，但 Denoiser 产生少量误差后，解码结果就会快速下降。

这也解释了为什么“重构准确率”不能单独作为 latent 好坏的标准。

## 7. TextLDM：可生成压力在表示侧：将 VAE + DiT 的两阶段结构用于文本

TextLDM 的路线非常干净：Transformer VAE 把离散 token 映射到连续 latent，DiT 在 latent 上做 flow matching，最后并行解码。它明确采用一 token 一 latent，不做长度压缩。

在表示塑形上，它同时使用 KL 和 REPA：前者约束分布，后者把 latent 对齐到 Qwen3-1.7B 的表示。整个框架接近图像生成中的 VAE + DiT 两阶段结构，只是将输入和输出从像素换成了 token。

我很想继续追它的 latent 宽度实验：TextLDM 报告 $d=64$ 最优，更宽的 128、192 反而变差。容量上去以后，Denoiser 的学习难度、噪声标定和 Decoder 的利用方式都会一起变，latent 不是越宽越好。

## 8. FlowMapLM：可生成压力在表示侧：不训练 Autoencoder，直接在 one-hot 空间学习 Flow

FlowMapLM 直接跳出了两阶段框架。它没有 Autoencoder，而是把 token 写成词表维度的 one-hot 向量，在这个连续空间上学习 flow，最后 argmax 回 token。

这样做的直接好处是不存在 Autoencoder 带来的表示接口错位：合法 token 是 simplex 的顶点，因此生成终点的几何定义非常明确。

它进一步学习 flow map，也就是不只学“此刻往哪走”，而是直接学“从时间 $t$ 跳到时间 $r$ 会到哪”。卖点就是少步，甚至一步生成。

代价是 one-hot width 等于 vocabulary size，空间维度很大，而且无法获得压缩 latent 的语义抽象。不过这也说明：**Autoencoder 并不是连续语言生成的必要条件，而是一种模型设计选择。**

## 9. LangFlow：可生成压力在表示侧：统一 CE 和 Flow Matching 的目标函数几何

LangFlow 同样没有独立 Autoencoder，直接在 token embedding 空间里做连续扩散。它把功夫花在了目标函数和噪声调度的几何上。

它用 Bregman divergence 把 token 交叉熵与 Flow Matching 放进一个统一框架；再从 information-uniform 的原则出发设计可学习的 Gumbel 调度器，让不同时间段承载的信息变化更均匀；最后明确加入 self-conditioning。

FlowMapLM 主要研究少步生成映射，LangFlow 则从目标函数和 noise schedule 的几何出发，并且同样不使用 Autoencoder。

## 九个模型可以分为四类

根据 Encoder、Denoiser 和 Decoder 分担的训练压力，可以把这些工作分成四类：

| 路线              | 代表模型               | 主要思路                                   |
| --------------- | ------------------ | -------------------------------------- |
| 表示优先            | LTF、Cosmos、TextLDM | 先把 latent 压好、塑好，生成器才能轻松                |
| 生成器优先           | AURORA-LM          | 信息别丢，复杂 latent 交给更强 denoiser           |
| 协同适配            | ELF、Cola-DLM、LDLM  | 联合优化表示和生成器                             |
| 不使用 Autoencoder | FlowMapLM、LangFlow | 直接在 token 的连续表示上生成，避免 Autoencoder 接口错位 |

这四条路线没有谁天然正确，因为它们优化的目标不同：

- 如果优先保证 reconstruction fidelity，可以采用 AURORA 式的高容量 latent。
- 想要一步或极少步生成，LTF、FlowMapLM 这类强结构约束更占便宜。
- 想要一套最简洁、最容易迁移的 recipe，ELF 和 TextLDM 更像工程基线。
- 如果希望 latent 直接适应 generation objective，可以研究 Cola-DLM 和 LDLM 的联合训练，但需要额外处理训练稳定性。

## 我们的 JEPA-CDLM 如何考虑这个问题？

如果只把我们的方法概括成“冻结 T5 + 4 倍压缩 + Flow Matching”，看起来只是选择了一个不激进的压缩率。看起来我们是把塑性压力给到了表示侧，我们实际想回答的问题更具体：**Autoencoder 学到的 latent 除了能够重构 token，是否也适合由 Flow 从噪声中生成？**

项目代码里这条方法叫 `jepa_ae`。它同时训练 representation learning 和 Flow generation，并让两个分支学习同一个 clean endpoint。

![JEPA-CDLM 当前训练与推理接口](/images/continuous-latent-lm/jepa-cdlm-project.svg)

### 问题一：我们选择多大的压缩率？

我们的起点是冻结的 T5-small。T5 先把 128 个 token 变成 128×512 的 contextual features，再交给可训练压缩器。当前重点 backend 是一个确定性的 TextVAE，名字虽然还叫 VAE，实际上没有 $\mu$、$\log\sigma$、重参数采样或 KL，更准确地说是 TextAE。

它把每 4 个 token feature 拼成一个 patch，因此无条件 LM1B 的 128-token canvas 会变成 32 个 latent slot。`tiny` 版本输出 512 维 latent，`small` 输出 768 维。Decoder 也是并行的，把 slot 展回 token-aligned T5 feature，再通过共享 unembed 还原文字。

这个压缩率位于 AURORA-LM 和 LTF 之间：

- 比 AURORA-LM 克制。它一 token 一 latent、宽度 768/1024，我们先把长度压 4 倍。
- 比 LTF 保守。它从 256 压到 8，压 32 倍；我们不要求 32 个 slot 承担整段文章的高度抽象。
- 跟 Cosmos 的 8 倍压缩也不同。我们先用 4 倍作为能诊断、能逐项消融的工作点，不急着拿压缩率当成绩。

这里还有一个容易被忽略的设计：condition 和 target 分开 patch，condition query 看不到 target，target query 可以看 condition 和全部 target。没有 condition 时，同一个模型自然退化成全 target 的双向网格。也就是说，翻译和无条件生成共用一套 TextAE，不需要藏两套 dataset-specific 架构。

### 问题二：JEPA 在这个模型中负责什么？

训练时有两个 compressor。

Online compressor 看到被加噪、置零或 mask 的 target feature，输出 $z_{\text{corrupt}}$；EMA compressor 读取 clean feature，慢慢更新，给出 stop-gradient 的 $z^*$。随后同一个 DiT 在 $t=1$ 扮演 JEPA predictor，把 corrupted online latent 预测回 $z^*$。

这条分支同时接三种监督：预测 latent 对齐 EMA teacher 的 JEPA MSE，Decoder hidden 对齐 T5 feature 的 MSE，以及最终 token CE。Flow Matching 则从 Gaussian noise 出发，也学习抵达同一个 $z^*$。推理时，ODE rollout 先生成 target latent，再过一次 $t=1$ predictor，最后由 TextAE Decoder 并行读回 token。

我们采用的判断标准是：

> 如果 clean latent 可以从受损表征中恢复，也可以由 Flow 从纯噪声生成，并且这两种输出都能被 Decoder 稳定读回 token，那么这个 latent space 才同时满足 representation 和 generation 的要求。

为了避免 online latent collapse，代码里还有 SigReg、VICReg 接口和运行时 latent restandardization；DiT 使用 x-prediction、self-conditioning 和独立的 decode final head。这些组件用于提高训练稳定性，但不能单独构成创新。JEPA-CDLM 需要证明的是：这套 teacher–student 结构能否让 latent 更适合生成，而不只是提高 reconstruction quality。

这里要把进度说清楚。确定性 TextAE、EMA teacher、JEPA terminal predictor、联合 Flow 和整套 endpoint probe 都已经在代码里；但目前最完整的生成实验主要来自前一版 Cosmos/Perceiver compressor 谱系，不是 TextAE 的最终成绩单。两种 backend 共用 `jepa_ae` 方法接口，所以旧实验能暴露 corruption、梯度和 readout 的结构性问题，却不能替新 TextAE 宣布胜利。

### 问题三：为什么 reconstruction 变好，generation 反而可能变差？

我们已经审计了 57 个 run-like 实验目录。其中一个反复出现的现象是：**reconstruction accuracy 提高时，generation quality 反而可能下降。**

| 配置 | real-token 重构 | Gen-PPL ↓ | MAUVE ↑ | 发生了什么 |
| --- | ---: | ---: | ---: | --- |
| 经典 LM1B，epoch 10 | 可用但不追求满分 | 110.50 | 0.8892 | raw T5 强腐蚀 + feature/Flow 双 bottleneck |
| Cosmos direct-768，2.5 epoch | acc≈94% | 1042.83 | 0.0174 | 噪声放在可训练投影后，模型走了高 SNR 捷径 |
| raw T5 pre-projection noise，2.5 epoch | acc≈58% | 234.55 | 0.5695 | 把噪声移到投影前，生成明显恢复 |
| token-LN + bottleneck + $\sigma=1$ | acc≈99.4% | 796.55 | 0.0938 | 名义相同的噪声，在归一化坐标里太弱 |
| token-LN + bottleneck + $\sigma=5$ | acc≈65% | 228.79 | 0.7953 | 更难重构，反而得到更好的生成覆盖 |

这些结果说明问题不只是“latent 应该使用 512 维还是 768 维”，还需要检查：**corruption 实际破坏了多少信息，以及模型能否通过 trainable projection 绕过 corruption。**

同一个 $\sigma=1$，加在标准差约 0.11 的 raw T5 feature 上，会产生较强的 corruption；加在 LayerNorm 后标准差约 1 的 feature 上，则只有大约 0 dB。如果先经过可训练投影再加噪，投影还可以放大信号尺度，使固定噪声的相对强度逐渐降低。此时 CE 会快速下降，Encoder gradient 增大，但 Flow loss 和 generation quality 会同时恶化。

这不是普通的“过拟合”。Online compressor、EMA clean endpoint、训练 corruption、Flow prediction 和真实 ODE terminal，正在形成几套互不兼容的坐标。

### 问题四：训练时的 readout input 和推理时是否一致？

现有 `jepa_ae` 默认用 $z_{\text{corrupt}}$ 训练 $t=1$ predictor 和 readout。CE、feature MSE、JEPA 都会沿着这条路更新 online compressor、共享 DiT trunk 和 Decoder；标准 Flow loss 在 EMA-teacher 配置下不会更新 compressor。

可推理时 Decoder 前面收到的是 ODE terminal。训练保证的是

$$
F_{t=1}(z_{\text{corrupt}}) \approx z^*,
$$

却没有直接保证

$$
F_{t=1}(z_{\text{ODE}}) \in \mathcal B_{\text{decoder}},
$$

其中 $\mathcal B_{\text{decoder}}$ 是 Decoder 能稳定读出正确 token 的区域。

项目中的 counterfactual probe 已经观察到一个明显的例子：某个 direct-768 checkpoint 的 $t=1$ readout 能把 training noisy latent 的 real-token CE 从 2.068 降到 0.030，却把 clean latent 的 CE 从 4.656 提高到 10.060，也把 Flow x-pred 的 CE 从 5.497 提高到 9.892。这说明 predictor 的确学到了映射，但映射后的 latent 不在 Decoder 对 clean latent 或 Flow prediction 有效的区域中。

因此，目前更需要解决的并不是增加一个 latent regularizer，而是让 readout 直接训练在生成过程实际会到达的 support 上。一个可以验证的方案是：让一部分 training row 使用 detached late-Flow prediction，例如 $t\in[0.7,1]$，其余 row 保留 noisy 和 clean anchor。这样可以避免 Decoder 直接处理早期的高噪声状态，同时减少 readout training 和 ODE inference 之间的 input mismatch。

代码已经有一个 `decode_from_flow_prediction` 的全开/全关实验口，但上面这种按概率混合、只选 late-$t$ 的稳定版本还没有成为默认方法。它仍然是待验证方案，不是已经拿到的贡献。

## 所以，这个项目到底有没有创新性？

我的判断是：**目前的方法有创新性候选，但还需要实验把“组件组合”和“新的机制”区分开。**

| 候选贡献 | 我的判断 | 可能的审稿质疑 | 需要什么证据 |
| --- | --- | --- | --- |
| 冻结 T5 + 4× TextAE + latent Flow | 工程价值明确，方法新颖性偏弱 | “LTF、Cosmos、TextLDM 都做过压缩 latent” | matched compute 下证明 4× 的速度/质量 Pareto 优于不压和 8×/32× |
| EMA teacher + JEPA predictor 塑形 compressed latent | 应用级新颖性中等，有潜力 | “BYOL/JEPA + denoising + Flow 的组合，机制并不新” | 同架构只把 $\lambda_{JEPA}$ 从 0 改到 1，生成质量和 endpoint geometry 都稳定改善 |
| 同一个 DiT 同时做 random-$t$ Flow 与 terminal JEPA prediction | 组合新颖性中等偏高 | “ELF/LDLM 已经联合训练，Cosmos 也做扰动恢复” | 证明共享 predictor 比分离网络或普通 denoising AE 更好，而不是参数量红利 |
| 统一 condition/target fixed-canvas TextAE 接口 | 架构整合干净，但更像系统贡献 | “这是 mask 和工程实现选择” | 在翻译、LM1B、XSum 上复用同一 backend，并给出严格 no-leak 与泛化结果 |
| 让 late-Flow/ODE landing support 直接进入 Decoder 训练 | 新颖性较高的方法候选 | “AURORA consistency、LDLM decoder noise 已经在处理 train–test gap” | 证明我们对齐的是 decoder-sensitive landing basin，并用 direct-vs-$t=1$、margin/CE 与生成指标建立因果链 |

我不会直接在论文摘要中声称“首次把 JEPA 用在 latent diffusion language model”。Cosmos 的 perturb-and-recover、EMA self-distillation，以及更广义的 JEPA/denoising 方法都可能与这个表述产生重叠。

更准确的定位是：**我们希望刻画 continuous latent language model 中的 Decoder-interface mismatch，并使用 generation-aware predictive representation learning 减少这种 mismatch。**JEPA 是实现方法，接口问题才是主要研究问题。如果 late-Flow readout alignment 没有效果，这个贡献也不能成立。

## 接下来需要优先验证哪些问题？

现在仓库里可调开关很多，但下一轮不该再一次改五六项。按“每次只改一个变量”的原则，我更想先回答下面这些问题：

| 问题 | 最小对照 | 它回答什么 |
| --- | --- | --- |
| 4× 压缩本身值不值 | ELF full-token → 零参数 patchify 4× | 只看长度压缩，不给 learned compressor 抢功 |
| 学习式 TextAE 有没有贡献 | patchify 4× → TextAE 4× | 固定 Flow 和容量，检查可学习表示是否真有用 |
| JEPA objective 有没有贡献 | 同一 `jepa_ae`，$\lambda_{JEPA}=0\to1$ | 避免把 EMA、联合训练、参数量一起混进“JEPA 提升” |
| EMA teacher 是否必要 | teacher=none → teacher=EMA | 稳定 teacher target 的收益是否超过 moving-target 代价 |
| landing 对齐是否有效 | corrupted-only readout → 25% late-Flow mix → 50% mix | 是否改善 ODE endpoint，而不只是降低训练 CE |
| 压缩率能否继续提高 | 最优协议下 $K=1\to4\to8$ | 等接口稳定后再谈速度–质量 Pareto |

所有候选先跑短轮，1 epoch 用于排除不收敛和明显坍缩；有希望的跑到 3 epoch 检查趋势，最后只留下 JEPA on/off、表现最好的 baseline 和 landing-aligned 版本跑足 10 epoch。结果接近时再补 3 个 seed，不根据单次的小差距形成结论。

评估不能只放 reconstruction CE。每个 checkpoint 至少要固定报告 1,024-sample Gen-PPL、MAUVE、unigram entropy、输出长度、empty count、real-token CE，以及 noisy / clean / matched-Flow / ODE terminal 在 direct Decoder 与 $t=1$ readout 下的成对变化。

为了避免根据单次结果不断修改方法，可以提前设置以下停止标准：

- 如果 TextAE 不能超过零参数 patchify，学习式压缩就很难作为主要贡献。
- $\lambda_{JEPA}=1$ 只让重构更好，却不改善 Gen-PPL、MAUVE 或 landing 指标，JEPA 就不能算主要贡献。
- late-Flow mix 降低训练 CE，却继续伤害 ODE endpoint，说明它仍然只适配了 training input support。
- 4× 在相同采样步数和计算预算下没有带来速度–质量优势，就别把压缩率写进标题。

这些停止标准会让创新性的判断更清楚。创新性不能只来自新的模块名称，需要先说明现有方法没有解决的问题，再通过控制变量实验验证提出的机制确实解决了这个问题。

## 总结：不同模型如何分配训练压力？

对比这九个模型以后，我认为目前还没有一种统一的 latent design。不同方法的区别，主要是如何在 Encoder、Denoiser 和 Decoder 之间分配训练压力。

以前大家问：连续 latent 能不能生成语言？

现在更好的问题是：

- 一条 latent 应该给每个 token 多少信息？
- 几何平滑该由 Encoder 保证，还是由 Denoiser 适应？
- 生成顺序和读出顺序分别要不要因果？
- Autoencoder 该冻结、联合训练，还是干脆删掉？
- 一步生成需要牺牲多少容量、连贯性和训练稳定性？

对于我们自己的项目，下一步不应该继续给某一个模块增加更多 trick，而是同时检查 latent capacity、noise calibration、Decoder robustness 和 train–inference consistency。只有当 Encoder 输出的表示、Denoiser 生成的表示，以及 Decoder 能够读取的表示落在相互兼容的区域中，实验才能说明这套方法解决了 continuous latent generation 的接口问题。

---

## 参考资料

- [AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling](https://arxiv.org/abs/2608.02602)
- [Latent Thought Flows](https://latent-thought-flows.vercel.app/)
- [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)
- [Cola-DLM: Continuous Latent Diffusion Language Model](https://arxiv.org/abs/2605.06548)
- [How to Train Your Latent Diffusion Language Model Jointly With the Latent Space](https://arxiv.org/abs/2605.07933)
- [Cosmos: Compressed and Smooth Latent Space for Text Diffusion Modeling](https://arxiv.org/abs/2506.21170)
- [TextLDM: Language Modeling with Continuous Latent Diffusion](https://arxiv.org/abs/2605.07748)
- [Flow Map Language Models: One-step Language Modeling via Continuous Denoising](https://arxiv.org/abs/2602.16813)
- [LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling](https://arxiv.org/abs/2604.11748)

> 注：这篇文章比较的是模型设计口径，不直接横比论文里的最终分数。不同工作的 tokenizer、数据规模、序列拼接、采样步数和 CFG 设置并不完全一致，把 benchmark 生排在一张榜上反而容易误导。
