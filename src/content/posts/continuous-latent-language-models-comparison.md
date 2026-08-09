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

## 那我们自己的 CDLM 站在哪？

我现在更倾向把自己的方案放在两个极端之间：用冻结 T5 提供稳定语义基座，外接可训练的长度压缩器，把长度压到 4 倍或 8 倍，再让全序列 Flow Matching 一次性生成 latent。

这个位置的好处是：不像 AURORA 那样把满长满宽的压力全交给 Denoiser，也不像 LTF 那样一口气压 32 倍、逼 Decoder 承担大量展开工作。

但它也暴露了接下来最该验证的三件事：

1. **容错解码。** Decoder 不能只见过完美的 $z_0$，还应该见过生成器实际会产出的带偏差 $\hat z_0$。
2. **宽度与 schedule 联动。** latent 宽度变了，noise schedule 不能照抄；AURORA 和 Cola-DLM 已经从两个方向给了证据。
3. **重构与可生成性分开评估。** 不只看 token reconstruction，还要看局部扰动后是否稳定、生成 latent 是否落在 Decoder 能读的邻域。

所以我们要设计的不是一个孤立的压缩器，而是一份三方都能履约的 latent 合同。

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
