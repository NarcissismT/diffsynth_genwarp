DA-Flow 的训练不是“让 diffusion model 直接回归光流”，而是**先把图像修复 diffusion model 训练成带时序注意力的特征提取器，再冻结它，用 RAFT 式光流网络在这些 diffusion 特征上训练光流头**。它的训练可以理解成两阶段。

## 1. 训练目标是什么

DA-Flow 要解决的问题是：输入两帧 degraded / low-quality 视频帧：

[
I^k_{LQ}, I^{k+1}_{LQ}
]

输出它们之间的 dense optical flow：

[
\hat f_{k\rightarrow k+1}=M(I^k_{LQ}, I^{k+1}_{LQ})
]

但是真实 degraded 视频没有 ground-truth flow，所以作者用 **HQ 视频帧对生成 pseudo ground-truth flow**，再用对应的 LQ 帧作为输入训练。也就是：

[
\text{input}: I^k_{LQ}, I^{k+1}_{LQ}
]

[
\text{supervision}: f^**{k\rightarrow k+1} = \text{SEA-RAFT}(I^k*{HQ}, I^{k+1}_{HQ})
]

论文在 introduction 和 loss 部分都说明：真实退化视频没有 GT flow，因此他们用 pretrained flow model 在 high-quality video 上生成 pseudo flow，然后把 degraded frames 喂给 DA-Flow 训练。

---

## 2. 第一阶段：训练 lifted diffusion model

DA-Flow 的 diffusion backbone 不是从零训练，而是基于 **DiT4SR** 这种图像修复 diffusion model。原始 DiT4SR 是逐帧处理图像，没有时序建模能力。作者做的关键改动是：把原来每帧独立的 MM-DiT attention 改成 **full spatio-temporal attention**。

原始图像 diffusion 是这样处理的：

[
F_m \in \mathbb{R}^{(BF)\times T \times C}
]

也就是把 frame 维度折叠到 batch 维度中，每一帧独立 attention。DA-Flow 改成：

[
\tilde F_m \in \mathbb{R}^{B\times (FT)\times C}
]

也就是让一个视频 clip 里的所有 frame token 都在同一个 attention 里交互。这样每个 token 可以 attend 到其他帧的 token，从而学习跨帧 correspondence。论文 Fig. 2(a) 也画得很清楚：Original MM-DiT 是 “Frame × Batch” 的 spatial attention，Lifted Full MM-DiT 变成 “Frame × Token” 的 spatio-temporal attention。

这一阶段的 loss 仍然是 diffusion / rectified flow 的训练目标。给定 LQ/HQ paired frame：

[
z^k_{LQ} = Enc(I^k_{LQ}),\quad z^k_{HQ}=Enc(I^k_{HQ})
]

然后构造 noisy latent：

[
z_t^k = (1-t)\epsilon + t z^k_{HQ}
]

网络预测 velocity：

[
v_t^k = D(z_t^k,t|z^k_{LQ})
]

监督目标是：

[
\mathcal{L}_{diff}
==================

\mathbb{E}*{k,t,\epsilon}
\left[
\left|
v_t^k - (z^k*{HQ}-\epsilon)
\right|_2^2
\right]
]

也就是说，第一阶段本质上还是训练一个**图像/视频修复 diffusion model**，只不过它被改造成可以跨帧 attention，因此其中间 Query/Key 特征会带有时序匹配能力。

实验设置上，第一阶段使用 **F=3 consecutive frames** 的视频 clip 训练 lifted diffusion model。训练数据来自 YouHQ，LQ 视频由 RealBasicVSR/Real-ESRGAN 风格的退化流程生成，包括 frame-level degradation 和 video compression。

---

## 3. 训练完第一阶段后，不直接输出光流，而是提取 diffusion 特征

这是 DA-Flow 最关键的一点：**diffusion model 不是最终光流预测器，而是 degradation-aware feature extractor。**

训练好的 lifted diffusion model 会在 denoising 过程中产生很多层的 attention Query/Key 特征。作者发现这些 Query/Key 特征天然有 correspondence 能力，所以取：

[
\tilde Q^k_{HQ},\quad \tilde K^{k+1}_{HQ}
]

其中 Query 来自第 (k) 帧，Key 来自第 (k+1) 帧。论文 Sec. 4.3 明确说，他们从 full spatio-temporal MM-Attention 层中提取 frame (k) 的 query feature 和 frame (k+1) 的 key feature。

作者还做了 feature analysis：直接用这些 Q/K feature 做 dot-product cost volume，再 softargmax 得到一个 zero-shot flow，用 pseudo GT flow 评估 EPE。这个分析用来选择哪些 diffusion 层最适合做 correspondence。最后 DA-Flow 选了四层：

[
{3, 13, 16, 17}
]

作为最终光流网络训练时的 diffusion feature source。

---

## 4. 第二阶段：冻结 diffusion model，训练 RAFT-style 光流网络

第二阶段才是真正训练光流预测器。

作者把第一阶段训练好的 lifted diffusion model (D_\phi) 冻结，然后训练一个基于 RAFT 的光流网络 (M_\theta)。整体结构是：

[
M_\theta = U \circ C \circ (\texttt{Up}(D_\phi), E)
]

其中：

* (D_\phi)：冻结的 lifted diffusion model，用来提取 degradation-aware Q/K/context 特征；
* (\texttt{Up})：DPT-based upsampling，把 diffusion feature 从 (H/16 \times W/16) 上采样到 (H/8 \times W/8)；
* (E)：普通 RAFT CNN encoder，用来补充细粒度局部纹理；
* (C)：RAFT 的 correlation operator；
* (U)：RAFT 的 iterative update operator。

论文明确说 DA-Flow 保留 RAFT 的 correlation operator 和 iterative update operator，同时加入 lifted diffusion model 和传统 CNN encoder。

---

## 5. 第二阶段的输入、监督和 loss

第二阶段的训练数据格式是：

[
(I^k_{LQ}, I^{k+1}*{LQ}, f^**{k\rightarrow k+1})
]

其中 (I^k_{LQ}, I^{k+1}_{LQ}) 是退化后的低质量帧，监督 (f^*) 不是人工 GT，而是：

[
f^*_{k\rightarrow k+1}
======================

\text{SEA-RAFT}(I^k_{HQ}, I^{k+1}_{HQ})
]

也就是用 SEA-RAFT 在对应的高质量帧对上生成 pseudo label。

训练 loss 是标准 RAFT 多迭代监督：

[
\mathcal{L}_{flow}
==================

\sum_{i=1}^{M}
\gamma^{M-i}
\left|
f^{(i)}_{k\rightarrow k+1}
--------------------------

f^*_{k\rightarrow k+1}
\right|_1
]

这里 (f^{(i)}) 是第 (i) 次 recurrent refinement 输出的 flow，越靠后的 iteration 权重越大。论文 Sec. 4.4 明确给出了这个 loss。

---

## 6. 特征如何进入 RAFT

DA-Flow 不是只用 diffusion feature，也不是只用 RAFT CNN feature，而是把两者 concat。

对第 (k) 帧：

[
F^k = \text{Concat}(F^k_{img}, F^{k,\uparrow}_{Q})
]

对第 (k+1) 帧：

[
F^{k+1} = \text{Concat}(F^{k+1}*{img}, F^{k+1,\uparrow}*{K})
]

context feature：

[
F^k_{h-ctx} = \text{Concat}(F^k_{ctx}, F^{k,\uparrow}_{ctx})
]

然后 (F^k) 和 (F^{k+1}) 构建 RAFT cost volume，(F^k_{h-ctx}) 送入 update operator。论文 Sec. 4.4 说明这些 hybrid feature 用于 cost volume construction 和 iterative update。

所以它的核心训练逻辑是：

> diffusion feature 提供 degradation-aware、结构性、跨帧对应先验；
> CNN feature 提供局部细节；
> RAFT recurrent update 负责最终的 dense flow refinement。

---

## 7. 具体训练配置

论文给出的主要配置是：

| 项目                    | 设置                                                         |
| --------------------- | ---------------------------------------------------------- |
| 训练阶段                  | 两阶段                                                        |
| Stage 1               | 训练 lifted diffusion model (D_\phi)                         |
| Stage 2               | 冻结 (D_\phi)，训练 flow pipeline (M_\theta)                    |
| 数据集                   | YouHQ                                                      |
| clip 长度               | (F=3) consecutive frames                                   |
| LQ 生成                 | RealBasicVSR / Real-ESRGAN degradation + video compression |
| pseudo GT             | SEA-RAFT on HQ frame pairs                                 |
| diffusion feature 层   | ({3,13,16,17})                                             |
| crop size             | (512 \times 512)                                           |
| batch size            | 32                                                         |
| GPU                   | 4 × NVIDIA H100                                            |
| learning rate         | (5\times10^{-5})                                           |
| training steps        | 两个阶段各 20K steps                                            |
| refinement iterations | 主文说训练 flow pipeline 用 (M=12) 次 refinement                  |

这些设置来自论文 Sec. 5.1。

---

## 8. 用一句话总结 DA-Flow 的训练

DA-Flow 的训练流程可以概括为：

> 先用 YouHQ 的 LQ/HQ 视频训练一个加入 full spatio-temporal attention 的 DiT4SR，让 diffusion 中间层 Q/K feature 学到跨帧对应；然后冻结这个 diffusion model，把它的 Q/K/context 特征上采样后与 RAFT CNN 特征拼接，用 SEA-RAFT 在 HQ 帧上生成的 pseudo flow 作为监督，训练 RAFT-style iterative optical flow network。

对你的任务来说，最值得借鉴的是第二阶段：**不要指望 diffusion model 直接 decode 出光流，而是把 diffusion 中间特征当作几何先验，再接一个 RAFT/FlowHead 式的可监督光流预测头。**
