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

最近这个方向有一种很典型的科研体验：论文一篇接一篇地冒出来，名字也一个比一个猛。AURORA-LM、LTF、ELF、Cola-DLM、TextLDM、FlowMapLM……每篇都像在另起炉灶。读到第五篇时，人已经开始怀疑自己：这真的是同一个领域吗？

> 如果你还不熟悉 Encoder、Decoder 和 latent space，可以先读[《Autoencoder 到底在做什么？》](/posts/autoencoder-explained/)。那篇用一个 Fashion-MNIST VAE 小项目把这三个词从头讲开，这篇默认你已经知道 Autoencoder 的基本数据流。

我把九个模型的原文摊开，硬塞进同一张表以后，反而发现了一件特别好玩的事：**它们基本都在搬同一块石头，只是选择把石头扔给不同的模块。**

- 有人说：先把 latent 压得又短又顺，生成器自然就轻松了。
- 有人说：别压，信息一丁点都别丢，给 denoiser 加装备就完事了。
- 有人说：Encoder、Denoiser、Decoder 别分家，三个人一起练。
- 还有人更干脆：为什么非要 Autoencoder？我直接在 one-hot 或 token embedding 上流。

所以这篇不按论文发布日期念摘要，也不比赛谁的 benchmark 高。我只追一个问题：每个模型认为困难在哪里，它又把主要的模型容量和训练技巧放在了哪里？

![连续潜空间语言模型的统一框架](/images/continuous-latent-lm/unified-framework.svg)

## 先说结论：Encoder–Denoiser–Decoder 能讲清，但要画成两条线

Encoder → Denoiser → Decoder 这个视角是对的，不过如果只画一条直线，很容易把训练和生成混在一起。更准确的读法是两段：

1. **先训练表示。** Encoder 把 token 变成干净 latent $z_0$，Decoder 再把它还原成 token。这里决定 latent 到底保留多少信息、空间是否平滑、Decoder 容不容错。
2. **再训练生成。** Denoiser 从噪声出发，学着生成同一种 latent。生成出的 $\hat z_0$ 最后交给 Decoder 读回文字。

中间那块 latent 其实是一份“合同”：Encoder 写，Denoiser 仿，Decoder 读。麻烦在于这三方的诉求天然打架。

- Decoder 希望 latent 信息越全越好，最好原文每个字都别丢。
- Denoiser 希望 latent 越规整越好，最好低维、平滑、没有奇怪尖角。
- 速度又希望所有位置并行生成，连贯性却偏偏喜欢有先后顺序。

理解了这个三角矛盾，后面九个模型就都不神秘了。

## 读论文前先把四个旋钮认清

### 旋钮一：到底压了什么？长度和宽度千万别混

设一段文字有 $L$ 个 token，Encoder 输出 $K$ 个 latent，每个 latent 有 $d$ 维。总容量粗略看就是 $K\times d$。

这里有两种完全不同的“压缩”：

- **压长度**：把 256 个 token 变成 8 个 latent，这是 LTF 的 32 倍压缩。
- **压宽度**：仍然一个 token 对一个 latent，但把每个 latent 从 512 维挤到 128 维，这是 ELF 的 bottleneck。

AURORA-LM 最激进的地方反而是**不压**：最强设置取 $K=L$，宽度还保留 768 或 1024。它相当于对 Decoder 说“信息管够”，然后转头对 Denoiser 说“剩下的你想办法”。

### 旋钮二：谁负责让 latent 变得“好生成”？

所谓 latent 好不好，不只看能不能把原文重构回来。还得再考一道：从噪声生成出来、带一点误差的 latent，Decoder 还能不能读？

常见办法有三类：

- **在表示侧塑形**：KL、masking、对齐冻结语言模型、扰动后恢复。
- **在生成侧补偿**：调 noise schedule、加输入瓶颈、consistency、self-conditioning。
- **让两侧共同适应**：Encoder / Denoiser / Decoder 联合训练。

这就是我说的“把石头扔给谁”。

### 旋钮三：latent 怎么生成，和 token 怎么读出，是两回事

这是整张表最容易踩的坑。

![latent 生成顺序与 token 读出顺序是两回事](/images/continuous-latent-lm/two-orders.svg)

LTF 可以一步把全部 latent 并行生成出来，但 Decoder 仍然一个 token 一个 token 地吐；AURORA-LM 则是 latent 按块生成，但 Query-based Decoder 不接收上一时刻吐出的 token，所以读回阶段可以一次前向并行完成。

判断一个 Decoder 是否自回归，别只看 attention mask 是不是 causal，要看**推理时是否必须把上一步输出的 token 再喂回去**。

### 旋钮四：训练完是冻结，还是大家继续一起动？

标准两阶段做法是 Autoencoder 训完冻结，再单独训练 latent generator。好处是稳、好诊断；坏处是 Denoiser 只能迁就一块不是为它量身定做的空间。

联合训练则允许 latent 跟着生成器一起变，但很容易出现合谋：Encoder 找到一套 Decoder 看得懂、Denoiser 却学不动的暗号，或者整个模型退化成低多样性的安全答案。LDLM 和 Cola-DLM 都在正面啃这块硬骨头。

## 九个模型，一张架构速查图

![九个连续文本生成模型架构速查](/images/continuous-latent-lm/model-atlas.svg)

下面正式逐个拆。每个模型我都只回答三句话：它觉得谁最该干活？它怎么干？代价是什么？

## 1. AURORA-LM：latent 不减肥，给 denoiser 办健身卡

如果只能用一句话概括 AURORA-LM，我会说：**它把 latent 的高容量完整保留下来，再把“让它可生成”的压力集中放到 denoiser。**

它的最佳设置基本不压长度，latent 宽度也很大。这样做的好处很直白：Decoder 不缺信息，token-level fidelity 容易保住。可坏处也一样直白：高维、满长的 latent 分布不是一个好学的对象。

于是 AURORA 给 denoiser 配了一套豪华工具箱：

- 用低秩入口限制 denoiser 读取 noisy latent 的通道，逼它抓结构而不是记噪声细节；
- 用与 latent 维度相关的 tan-$d$ noise schedule，把训练预算重新分配；
- 用 self-trajectory consistency 缩小训练时“独立噪声档”和推理时“连续轨迹”之间的裂缝；
- 再加 self-conditioning，把自己上一步的预测喂回来。

注意，这不等于 Autoencoder 侧完全没做事。它仍然有 token embedding dropout、latent dropout 和 Query-based Decoder。更准确的说法是：**AURORA 不愿意靠压缩换一个简单空间，所以把最有辨识度的“塑性”设计放在了 denoiser 一侧。**

这套哲学像什么？别人为了让车好开，先把路修直；AURORA 说路先别动，我给车加四驱、差速锁和主动悬挂。

## 2. LTF：先把 256 个 token 塞成 8 个想法，再一步生成

LTF 站在 AURORA 的另一个极端：**latent 不但要压，而且要往狠了压。**256 个 token 只留下 8 个 latent，长度直接压 32 倍。

这么压最怕什么？Encoder 偷懒，把局部字面信息塞进 latent；Decoder 也偷懒，只靠前文猜下一个词。LTF 的办法是两边一起施压：Encoder 训练时真的丢 token，Decoder 对前文的访问也会被随机切断，再配一个滑动窗口。意思很明确：谁都别抄答案，latent 必须学会概括。

第二阶段用 MeanFlow，一步生成全部 latent，速度非常漂亮。但别忘了上一节的坑：**它的 latent 是并行出的，token 却还是由自回归 Decoder 串行读回。**

所以 LTF 的赌注是：把长文本先压成少量“想法”，想法可以一口气生成；至于把想法展开成句子，仍然交给擅长连贯写作的自回归机制。

## 3. ELF：别造一座新城，先在 T5 的地基上开流

ELF 的气质是“能少改就少改”。它直接拿冻结的 T5 encoder 表示起手，不压序列长度，只通过线性 bottleneck 把 512 维降到 128 维。

更有意思的是，它默认没有一套泾渭分明的独立 Decoder：同一个网络大部分时间做 flow matching，小部分时间在 $t=1$ 的端点做 token 解码。一个网络轮流扮演 Denoiser 和 Decoder，训练配比大致是 80% MSE、20% CE。

ELF 还比较了联合训练和标准两阶段冻结，结果的 trade-off 接近。它选择联合方案，主要是流程更省，而且能把生成 PPL 往更低处推。

ELF 对 latent 没有太强的发明欲。它的判断很务实：成熟 embedding 本身已经够用，连续 flow 做一点适配就能工作。

## 4. Cola-DLM：VAE 和 DiT 别相亲了，直接一起过日子

Cola-DLM 用的是一条很规整的路线：严格因果 VAE 负责 latent，DiT + Flow Matching 负责生成，latent 不压长度，宽度从 16、64 到 128 都系统扫过。

它在表示侧用 KL 和 BERT 式 masking。KL 让空间别长得太野，masking 则防止 Encoder 只复制表面 token、不学语义。

Cola 最该单独记住的有两点。

第一，它让 VAE 和 DiT 联合微调，不把 VAE 训完就永远冻住。第二，它发现最优 timeshift 会随着 latent 维度变化，而且不只报告经验现象，还给了理论解释。同样的噪声加在 16 维和 128 维空间里，破坏程度根本不是一回事，noise schedule 不能脱离维度来谈。

它的 latent 生成也是 block-causal：块间有先后，块内并行。质量和速度之间不再只有“全并行”与“全串行”两个档。

## 5. LDLM：三个人一起练可以，但千万别直接开跑

LDLM 是这批模型里对“联合训练”最较真的一个：Encoder、Diffusion Model、Decoder 全部一起动。它希望可训练 Encoder 把预训练语言模型的表示重新塑造成一块既好去噪、又好解码的 latent 空间。

理想很美，作者也很诚实：naive joint training 会崩，生成多样性会变得很低。于是他们给出一套补救配方，包括 MSE decoder loss、diffusion-to-encoder warmup、自适应时间步采样，以及给 Decoder 输入加噪。

这里我记下的不是某一个 trick，而是一条很不舒服、又绕不过去的经验：允许 latent 适应 Denoiser，不代表它自然就会适应。训练日程和损失没配好，联合优化很容易找到一个三方损失都好看、生成质量却很差的投机解。

## 6. Cosmos：重构满分没用，latent 还得经得住扩散

Cosmos 关心的是两个词：compressed 和 smooth。它从冻结的 BERT 类 Encoder 起步，最多把长度压到 8 倍，同时让 Autoencoder 的表示对齐冻结语言模型的内部激活。

它还做 perturb-and-recover：对 hidden state 随机置零或加噪，再要求恢复干净表示。这不是显式地罚 Jacobian 或 Lipschitz 常数，但效果上是在逼 latent 流形对小扰动更稳。

Cosmos 给整个领域提了个醒：token 能 100% 重构，不代表 diffusion 就能学。Autoencoder 可能把信息藏在极其脆弱、极其弯曲的坐标里；Decoder 当然读得出来，可 Denoiser 只要偏半步就掉下悬崖。

这也解释了为什么“重构准确率”不能单独作为 latent 好坏的标准。

## 7. TextLDM：把图像 latent diffusion 的标准答案搬到文本

TextLDM 的路线非常干净：Transformer VAE 把离散 token 映射到连续 latent，DiT 在 latent 上做 flow matching，最后并行解码。它明确采用一 token 一 latent，不做长度压缩。

在表示塑形上，它同时用了 KL 和 REPA：前者约束分布，后者把 latent 对齐到 Qwen3-1.7B 的表示。整个框架很像图像生成里已经成熟的 VAE + DiT 两阶段范式，只是把像素换成了语言。

我很想继续追它的 latent 宽度实验：TextLDM 报告 $d=64$ 最优，更宽的 128、192 反而变差。容量上去以后，Denoiser 的学习难度、噪声标定和 Decoder 的利用方式都会一起变，latent 不是越宽越好。

## 8. FlowMapLM：我连 latent 都不学，直接在 one-hot 上流

FlowMapLM 直接跳出了两阶段框架。它没有 Autoencoder，而是把 token 写成词表维度的 one-hot 向量，在这个连续空间上学习 flow，最后 argmax 回 token。

这样做最大的好处是没有“Encoder 写了一种暗号、Decoder 才看得懂”的接口错位：合法 token 就是 simplex 的顶点，终点几何非常明确。

它进一步学习 flow map，也就是不只学“此刻往哪走”，而是直接学“从时间 $t$ 跳到时间 $r$ 会到哪”。卖点就是少步，甚至一步生成。

代价也摆在桌面上：one-hot 宽度等于词表大小，空间巨大，而且你放弃了压缩 latent 带来的语义抽象。但它证明了一件事：**Autoencoder 不是连续语言生成的入场券，只是一种设计选择。**

## 9. LangFlow：既然 CE 和 Flow Matching 都在教 token，干脆统一几何

LangFlow 同样没有独立 Autoencoder，直接在 token embedding 空间里做连续扩散。它把功夫花在了目标函数和噪声调度的几何上。

它用 Bregman divergence 把 token 交叉熵与 Flow Matching 放进一个统一框架；再从 information-uniform 的原则出发设计可学习的 Gumbel 调度器，让不同时间段承载的信息变化更均匀；最后明确加入 self-conditioning。

如果说 FlowMapLM 是从“少步映射”绕开 Autoencoder，LangFlow 就是从“目标函数几何”绕开 Autoencoder。

## 把九个模型重新分组，其实只有四条路

看完以后，可以把这批工作压成四种研究哲学：

| 路线 | 代表模型 | 核心信念 |
| --- | --- | --- |
| 表示优先 | LTF、Cosmos、TextLDM | 先把 latent 压好、塑好，生成器才能轻松 |
| 生成器优先 | AURORA-LM | 信息别丢，复杂 latent 交给更强 denoiser |
| 协同适配 | ELF、Cola-DLM、LDLM | 表示与生成器要一起磨合，不能各练各的 |
| 绕开 Autoencoder | FlowMapLM、LangFlow | 直接在 token 的连续表示上生成，消灭接口错位 |

这四条路线没有谁天然正确，因为它们优化的目标不同：

- 想要最高重构保真，AURORA 式高容量 latent 很诱人。
- 想要一步或极少步生成，LTF、FlowMapLM 这类强结构约束更占便宜。
- 想要一套最简洁、最容易迁移的 recipe，ELF 和 TextLDM 更像工程基线。
- 想让 latent 真正为生成服务，Cola-DLM 和 LDLM 的联合训练更值得深挖，但训练风险也最高。

## 那我们自己的 JEPA-CDLM 到底在想什么？

上一篇写到这里时，我只把自己的方案概括成“冻结 T5 + 4 倍压缩 + Flow Matching”。现在回头看，这句话太轻了，听上去只是选了一个不激进的压缩率。代码里真正想回答的问题其实更具体：**latent 不该只是 Autoencoder 重构任务的副产品，它得从出生那一刻起就为生成服务。**

项目代码里这条方法叫 `jepa_ae`。它没有完全押在 Encoder，也没有把锅全甩给 Denoiser，而是试图让表示学习和生成学习从第一步就围着同一个 clean endpoint 转。

![JEPA-CDLM 当前训练与推理接口](/images/continuous-latent-lm/jepa-cdlm-project.svg)

### 我们先站在两个极端中间

我们的起点是冻结的 T5-small。T5 先把 128 个 token 变成 128×512 的 contextual features，再交给可训练压缩器。当前重点 backend 是一个确定性的 TextVAE，名字虽然还叫 VAE，实际上没有 $\mu$、$\log\sigma$、重参数采样或 KL，更准确地说是 TextAE。

它把每 4 个 token feature 拼成一个 patch，因此无条件 LM1B 的 128-token canvas 会变成 32 个 latent slot。`tiny` 版本输出 512 维 latent，`small` 输出 768 维。Decoder 也是并行的，把 slot 展回 token-aligned T5 feature，再通过共享 unembed 还原文字。

这让我们的容量位置正好夹在两头：

- 比 AURORA-LM 克制。它一 token 一 latent、宽度 768/1024，我们先把长度压 4 倍。
- 比 LTF 保守。它从 256 压到 8，压 32 倍；我们不要求 32 个 slot 承担整段文章的高度抽象。
- 跟 Cosmos 的 8 倍压缩也不同。我们先用 4 倍作为能诊断、能逐项消融的工作点，不急着拿压缩率当成绩。

这里还有一个容易被忽略的设计：condition 和 target 分开 patch，condition query 看不到 target，target query 可以看 condition 和全部 target。没有 condition 时，同一个模型自然退化成全 target 的双向网格。也就是说，翻译和无条件生成共用一套 TextAE，不需要藏两套 dataset-specific 架构。

### JEPA 在这里不是装饰，它负责固定靶心

训练时有两个 compressor。

Online compressor 看到被加噪、置零或 mask 的 target feature，输出 $z_{\text{corrupt}}$；EMA compressor 读取 clean feature，慢慢更新，给出 stop-gradient 的 $z^*$。随后同一个 DiT 在 $t=1$ 扮演 JEPA predictor，把 corrupted online latent 预测回 $z^*$。

这条分支同时接三种监督：预测 latent 对齐 EMA teacher 的 JEPA MSE，Decoder hidden 对齐 T5 feature 的 MSE，以及最终 token CE。Flow Matching 则从 Gaussian noise 出发，也学习抵达同一个 $z^*$。推理时，ODE rollout 先生成 target latent，再过一次 $t=1$ predictor，最后由 TextAE Decoder 并行读回 token。

我们的想法其实很朴素：

> 如果 clean latent 既能被受损表征预测，又能被 Flow 从纯噪声生成，还能被 Decoder 稳定读回 token，那么它才算是一块合格的生成空间。

为了防止 online latent 塌成常量，代码里还有 SigReg、VICReg 接口和运行时 latent restandardization；DiT 使用 x-prediction、self-conditioning 和独立的 decode final head。这些组件并不自动构成创新，它们更像护栏。JEPA-CDLM 真正要证明的是：这套 teacher–student 结构有没有把 latent 塑造成“更容易生成”的空间，而不只是“更容易重构”的空间。

这里要把进度说清楚。确定性 TextAE、EMA teacher、JEPA terminal predictor、联合 Flow 和整套 endpoint probe 都已经在代码里；但目前最完整的生成实验主要来自前一版 Cosmos/Perceiver compressor 谱系，不是 TextAE 的最终成绩单。两种 backend 共用 `jepa_ae` 方法接口，所以旧实验能暴露 corruption、梯度和 readout 的结构性问题，却不能替新 TextAE 宣布胜利。

### 然后实验给我们泼了一盆很值钱的冷水

我们已经审计了 57 个 run-like 实验目录。最反直觉、也最有价值的发现是：**重构做得过于漂亮，往往正是生成正在变坏的信号。**

| 配置 | real-token 重构 | Gen-PPL ↓ | MAUVE ↑ | 发生了什么 |
| --- | ---: | ---: | ---: | --- |
| 经典 LM1B，epoch 10 | 可用但不追求满分 | 110.50 | 0.8892 | raw T5 强腐蚀 + feature/Flow 双 bottleneck |
| Cosmos direct-768，2.5 epoch | acc≈94% | 1042.83 | 0.0174 | 噪声放在可训练投影后，模型走了高 SNR 捷径 |
| raw T5 pre-projection noise，2.5 epoch | acc≈58% | 234.55 | 0.5695 | 把噪声移到投影前，生成明显恢复 |
| token-LN + bottleneck + $\sigma=1$ | acc≈99.4% | 796.55 | 0.0938 | 名义相同的噪声，在归一化坐标里太弱 |
| token-LN + bottleneck + $\sigma=5$ | acc≈65% | 228.79 | 0.7953 | 更难重构，反而得到更好的生成覆盖 |

这几组实验把问题从“latent 该用 512 还是 768 维”改写成了更靠谱的版本：**corruption 到底破坏了多少信息，模型又能不能绕过它？**

同一个 $\sigma=1$，加在标准差约 0.11 的 raw T5 feature 上，是一次相当凶的破坏；加在 LayerNorm 后标准差约 1 的 feature 上，只剩下大约 0 dB。更糟的是，如果先经过可训练投影再加噪，投影可以把信号尺度放大，让固定噪声越来越像摆设。CE 会飞快下降，Encoder 梯度越来越大，Flow loss 和生成质量却一起恶化。

这不是普通的“过拟合”。Online compressor、EMA clean endpoint、训练 corruption、Flow prediction 和真实 ODE terminal，正在形成几套互不兼容的坐标。

### 当前最大的洞：Decoder 学会接一种球，比赛却传来另一种球

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

项目里的反事实 probe 已经见过最坏情况：某个 direct-768 checkpoint 的 $t=1$ readout 能把训练 noisy latent 的 real-token CE 从 2.068 降到 0.030，却把 clean latent 的 CE 从 4.656 推坏到 10.060，也把 Flow x-pred 的 CE 从 5.497 推坏到 9.892。它不是没有学会收缩，而是收缩到了错误的私人地址。

这也是我现在对整个项目最明确的判断：我们缺的不是再加一个 latent regularizer，而是让 readout 直接覆盖生成时真正会到达的 support。比较自然的下一步，是让一部分训练 row 使用 detached late-Flow prediction，例如 $t\in[0.7,1]$，剩余 row 继续保留 noisy 和 clean anchor。这样既不把早期高噪声状态硬塞给 Decoder，也能逐步关上训练与推理的接口缝。

代码已经有一个 `decode_from_flow_prediction` 的全开/全关实验口，但上面这种按概率混合、只选 late-$t$ 的稳定版本还没有成为默认方法。它仍然是待验证方案，不是已经拿到的贡献。

## 所以，这个项目到底有没有创新性？

短答案：**有研究创新的胚子，但当前代码堆叠本身还不够成为一个强 novelty claim。**

| 候选贡献 | 我的判断 | 审稿人最可能怎么打 | 需要什么证据才能站住 |
| --- | --- | --- | --- |
| 冻结 T5 + 4× TextAE + latent Flow | 工程价值明确，方法新颖性偏弱 | “LTF、Cosmos、TextLDM 都做过压缩 latent” | matched compute 下证明 4× 的速度/质量 Pareto 优于不压和 8×/32× |
| EMA teacher + JEPA predictor 塑形 compressed latent | 应用级新颖性中等，有潜力 | “BYOL/JEPA + denoising + Flow 的组合，机制并不新” | 同架构只把 $\lambda_{JEPA}$ 从 0 改到 1，生成质量和 endpoint geometry 都稳定改善 |
| 同一个 DiT 同时做 random-$t$ Flow 与 terminal JEPA prediction | 组合新颖性中等偏高 | “ELF/LDLM 已经联合训练，Cosmos 也做扰动恢复” | 证明共享 predictor 比分离网络或普通 denoising AE 更好，而不是参数量红利 |
| 统一 condition/target fixed-canvas TextAE 合同 | 架构整合干净，但更像系统贡献 | “这是 mask 和工程实现选择” | 在翻译、LM1B、XSum 上复用同一 backend，并给出严格 no-leak 与泛化结果 |
| 让 late-Flow/ODE landing support 直接进入 Decoder 训练 | 目前最强的方法创新候选 | “AURORA consistency、LDLM decoder noise 已经在处理 train–test gap” | 证明我们对齐的是 decoder-sensitive landing basin，并用 direct-vs-$t=1$、margin/CE 与生成指标建立因果链 |

我不会把“首次把 JEPA 用在 latent diffusion language model”直接写进论文摘要。这个说法太容易被 Cosmos 的 perturb-and-recover、普通 EMA self-distillation，以及更广义的 JEPA/denoising 文献击穿。

更稳的定位是：**我们发现并刻画了连续 latent language model 的 Decoder-interface mismatch，再用 generation-aware predictive representation learning 去关闭它。**JEPA 是工具，接口问题才是论文故事。如果 late-Flow readout 对齐最后无效，那也应该老实放弃这条 claim，而不是靠换名字保住它。

## 接下来最重要的不是多跑，而是把因果拆干净

现在仓库里可调开关很多，但下一轮不该再一次改五六项。按“每次只改一个变量”的原则，我更想先回答下面这些问题：

| 问题 | 最小对照 | 它回答什么 |
| --- | --- | --- |
| 4× 压缩本身值不值 | ELF full-token → 零参数 patchify 4× | 只看长度压缩，不给 learned compressor 抢功 |
| 学习式 TextAE 有没有贡献 | patchify 4× → TextAE 4× | 固定 Flow 和容量，检查可学习表示是否真有用 |
| JEPA objective 有没有贡献 | 同一 `jepa_ae`，$\lambda_{JEPA}=0\to1$ | 避免把 EMA、联合训练、参数量一起混进“JEPA 提升” |
| EMA teacher 是否必要 | teacher=none → teacher=EMA | 稳定靶心的收益是否超过 moving-target 代价 |
| landing 对齐是否治到根上 | corrupted-only readout → 25% late-Flow mix → 50% mix | 是否真正改善 ODE endpoint，而不是继续刷训练 CE |
| 压缩率能否继续提高 | 最优协议下 $K=1\to4\to8$ | 等接口稳定后再谈速度–质量 Pareto |

筛选也要克制。所有候选先跑短轮，1 epoch 只负责排除不收敛和明显坍缩；有希望的跑到 3 epoch 看趋势，最后只留下 JEPA on/off、最强 baseline 和 landing-aligned 版本跑足 10 epoch。结果接近时再补 3 个 seed，不拿单次小差距讲故事。

评估不能只放 reconstruction CE。每个 checkpoint 至少要固定报告 1,024-sample Gen-PPL、MAUVE、unigram entropy、输出长度、empty count、real-token CE，以及 noisy / clean / matched-Flow / ODE terminal 在 direct Decoder 与 $t=1$ readout 下的成对变化。

这里还有几条很干脆的 kill rule：

- TextAE 赢不了零参数 patchify，说明“学一个压缩空间”的故事很弱。
- $\lambda_{JEPA}=1$ 只让重构更好，却不改善 Gen-PPL、MAUVE 或 landing 指标，JEPA 就不能算主要贡献。
- late-Flow mix 降低训练 CE，却继续伤害 ODE endpoint，说明我们又造了一个私人 basin。
- 4× 在相同采样步数和计算预算下没有带来速度–质量优势，就别把压缩率写进标题。

我反而觉得这种“随时准备杀死自己想法”的状态很健康。创新不是从模块名字里找出来的，是从一个别人没有解释清楚的失败里长出来，再由一组谁都挑不出混变量的实验保下来。

## 最后：这个领域不是在找一种 latent，而是在分配困难

九篇论文读完，我最大的感受不是“连续语言模型终于有标准答案了”，恰恰相反：这个领域刚刚开始学会把问题拆对。

以前大家问：连续 latent 能不能生成语言？

现在更好的问题是：

- 一条 latent 应该给每个 token 多少信息？
- 几何平滑该由 Encoder 保证，还是由 Denoiser 适应？
- 生成顺序和读出顺序分别要不要因果？
- Autoencoder 该冻结、联合训练，还是干脆删掉？
- 一步生成需要牺牲多少容量、连贯性和训练稳定性？

我现在最想盯的，不是再给某一个模块堆两个新 trick，而是把 latent 容量、噪声标定、Decoder 容错和训练/推理一致性放在一起设计。Encoder、Denoiser、Decoder 单独拿出来都讲不完这个故事，最后还得看它们中间那份合同能不能履约。

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
