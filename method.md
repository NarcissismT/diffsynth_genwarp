你的判断是对的：不要让 diffusion 最终“生成拉直后的图片”，而应该让模型预测几何位移场，再用这个位移场去采样原始扭曲图像。 这样最后图像里的文字像素来自原图，而不是 diffusion/VAE decoder 重新生成，所以可以避免文字被篡改、变糊、变形。
更适合你的方向可以概括成：
[
I_{warp}
\rightarrow
\text{Diffusion feature prior + CNN local feature}
\rightarrow
\hat{F}{rect \rightarrow warp}
\rightarrow
\text{grid_sample}(I{warp}, \hat{F})
\rightarrow
\hat{I}_{rect}
]
这里的 diffusion 不负责输出 RGB 图，而是负责提供“页面结构、弯曲趋势、全局几何”的先验；光流/位移场负责真正的像素搬运。
1. 先明确：你的任务不是传统 optical flow，而是 document dewarping flow
DA-Flow 原文做的是两帧 degraded video 之间的 optical flow。它的核心做法是：把 restoration diffusion model 的中间特征变成 degradation-aware correspondence feature，然后和 CNN feature 融合，再沿用 RAFT 的 correlation volume 和 iterative update 来预测光流。RAFT 本身也是通过 per-pixel feature、4D correlation volume 和 recurrent update 逐步更新 flow field。(CVLab)
但你的输入只有一张弯曲图像，目标是输出一个把它拉直的位移场。因此你的 flow 更准确地说是：
[
F_{rect \rightarrow warp}
]
也就是：对于 rectified image 上的每个像素位置，去 warped image 的哪个位置采样。
最终恢复时最好使用 backward warping：
[
\hat{I}_{rect}(x,y)
I_{warp}\big(x+\hat{u}(x,y),\ y+\hat{v}(x,y)\big)
]
这类“预测位移场并通过几何变换拉直文档”的思路本身在 document dewarping 里是成立的，已有工作直接训练网络回归 pixel-wise displacement，并用位移场完成文档矫正，同时加入局部平滑约束来稳定位移场。(arXiv)
2. 为什么不能直接用 Qwen Image Edit 的输出作为最终结果？
Qwen-Image-Edit 确实很强，它是基于 Qwen-Image 的图像编辑模型，并且官方说明它同时把输入图像送入 Qwen2.5-VL 做语义控制、送入 VAE Encoder 做外观控制。(Hugging Face) 但这也解释了你观察到的问题：只要走 diffusion/VAE 的图像生成路径，文字就有可能被重新解释、重新绘制，而不是严格保留原始像素。
所以你的系统里应该把 Qwen Image Edit 放在“辅助几何理解”的位置，而不是“最终产图”的位置。
也就是说：
错误路线：
[
I_{warp}
\rightarrow
\text{Qwen Image Edit}
\rightarrow
I_{rect}^{qwen}
]
推荐路线：
[
I_{warp}
\rightarrow
\text{Qwen diffusion hidden features}
\rightarrow
\hat{F}
\rightarrow
\text{warp original pixels}
\rightarrow
\hat{I}_{rect}
]
这和最近 document dewarping diffusion 工作的趋势也是一致的。比如 DvD 明确指出，直接把 diffusion 用在高分辨率复杂文档图像上会面临不忠实控制问题，因此它改为做 coordinate-level denoising，也就是生成几何映射，而不是生成像素图像。(arXiv)
3. 我建议你的最终模型结构
推荐主方案：Diffusion Feature Enhanced Flow Network
整体结构可以设计成：
Input warped image I_w
        │
        ├── CNN Local Encoder
        │       提取文字边缘、笔画、局部纹理、纸张边界
        │
        ├── Frozen Qwen-Image-Edit Feature Encoder
        │       提取页面结构、弯曲趋势、全局语义布局
        │       不使用它最终生成的 RGB 图像
        │
        ├── Feature Fusion
        │       CNN feature + diffusion hidden feature + coordinate encoding
        │
        ├── Flow Decoder / RAFT-style Recurrent Refinement
        │       输出 backward displacement field
        │
        └── grid_sample
                用预测 flow 从原始扭曲图像采样
最终输出不是图像，而是：
[
\hat{F}_{rect \rightarrow warp}
\in
\mathbb{R}^{H \times W \times 2}
]
然后：
[
\hat{I}_{rect}
\text{GridSample}(I_{warp},\ \text{Id}+\hat{F}_{rect \rightarrow warp})
]
这样恢复图像中的文字来自原始图片，不来自 diffusion 重新生成。
4. 训练数据应该怎么组织？
你现在已经有三种东西：
扭曲图像
[
I_{warp}
]
拉直后的正常图像
[
I_{rect}
]
从扭曲图像恢复到正常图像的 flow
[
F_{gt}
]
最推荐把 flow 统一成 backward flow：
[
F_{gt}^{rect \rightarrow warp}
]
含义是：rectified image 上每一个位置应该去 warped image 的哪里采样。
训练样本最好保存成：
{
  warped_image: I_warp,
  rectified_image: I_rect,
  backward_flow: F_rect_to_warp,
  valid_mask: M
}
其中 valid_mask 用来标记哪些 rectified pixels 能够在 warped image 中找到有效来源。纸张外部、黑边、采样越界区域都应该 mask 掉。
5. 损失函数应该怎么设计？
不要只训练 flow，也不要只训练 image reconstruction。推荐组合如下：
5.1 Flow supervision loss
因为你已经有 ground-truth flow，所以这是主监督：
[
\mathcal{L}_{flow}
\left|
\hat{F} - F_{gt}
\right|_1
]
或者用 Charbonnier loss：
[
\mathcal{L}_{flow}
\sqrt{
\left(\hat{F} - F_{gt}\right)^2 + \epsilon^2
}
]
这个 loss 直接约束模型学到正确几何。
5.2 Warping reconstruction loss
用预测 flow 去拉直原图：
[
\hat{I}_{rect}
\text{GridSample}(I_{warp}, \hat{F})
]
然后和真实拉直图像比较：
[
\mathcal{L}_{img}
\left|
M \odot
(\hat{I}{rect} - I{rect})
\right|_1
]
这个 loss 的作用是保证 flow 不只是数值接近，而是真的能把图像拉直。
5.3 Smoothness loss
文档纸张的形变通常是连续的，所以 flow 不应该有剧烈跳变：
[
\mathcal{L}_{smooth}
\left|
\nabla_x \hat{F}
\right|_1
+
\left|
\nabla_y \hat{F}
\right|_1
]
可以做 edge-aware smoothness，让文字边缘附近不要被过度平滑：
[
\mathcal{L}_{smooth}
e^{-\alpha |\nabla I|}
\cdot
|\nabla \hat{F}|
]
这样可以避免页面整体弯曲场出现毛刺，同时保留文字边缘附近的局部精度。
5.4 Fold / Jacobian regularization
为了避免预测出来的 flow 出现折叠、自交、局部翻转，可以加 Jacobian 约束：
[
\mathcal{L}_{fold}
\max(0, -\det(J_{\hat{F}}))
]
这对文档矫正很重要，因为如果 flow 局部折叠，文字会被撕裂或者重叠。
5.5 总损失
可以先用这个组合：
[
\mathcal{L}
\lambda_1 \mathcal{L}{flow}
+
\lambda_2 \mathcal{L}{img}
+
\lambda_3 \mathcal{L}{smooth}
+
\lambda_4 \mathcal{L}{fold}
]
初始阶段可以设：
λ_flow   = 1.0
λ_img    = 0.5
λ_smooth = 0.01
λ_fold   = 0.05
训练稳定后再调。
6. Qwen Image Edit 应该怎么接入？
这里有三种接入方式，我建议优先级如下。
方案 A：只用 Qwen 的 hidden features，不用它的输出图像
这是最推荐的。
流程：
I_warp
  → Qwen-Image-Edit encoder / DiT blocks
  → extract intermediate features
  → DPT/FPN upsample
  → concatenate with CNN features
  → flow decoder
  → output displacement field
重点是：不要拿 Qwen 生成的 rectified RGB 图作为最终结果。
你可以提取的特征包括：
middle DiT block hidden states
attention Q / K / V features
prompt-conditioned image tokens
multi-scale latent features
然后通过一个 adapter 变成适合 flow decoder 的特征：
[
F_{diff}
\text{Adapter}
(
\text{QwenHidden}(I_{warp}, prompt)
)
]
再和 CNN feature 融合：
[
F_{hybrid}
\text{Concat}
(
F_{cnn}, F_{diff}, E_{coord}
)
]
其中 (E_{coord}) 是二维位置编码，因为 dewarping 强依赖绝对位置。
这个方案最接近 DA-Flow 的精神：diffusion 提供强先验，CNN 保留局部细节，最终由 flow head 输出几何位移场。 DA-Flow 本身也是融合 diffusion feature 和 conventional CNN feature，而不是用 diffusion 直接生成光流图像。(CVLab)
方案 B：让 diffusion 直接生成 flow，而不是生成 RGB 图
这是更有研究价值的方案，但实现难度更高。
把 flow 当作 diffusion 的生成目标：
[
F_t
\sqrt{\bar{\alpha}t}F{gt}
+
\sqrt{1-\bar{\alpha}_t}\epsilon
]
模型输入：
condition: warped image I_warp
noisy target: F_t
timestep: t
output: noise ε 或 clean flow F_0
也就是训练一个 conditional diffusion model：
[
p(F \mid I_{warp})
]
最终：
I_warp → diffusion flow generator → F_hat → grid_sample → I_rect_hat
这个方案的优点是非常符合你的目标：diffusion 不再生成文字，而是生成平滑、结构合理的位移场。DvD 的 coordinate-level denoising 思路就是这类方向：用 diffusion 生成几何映射，而不是直接生成像素。(arXiv)
缺点是：如果你直接用 Qwen Image Edit 来做这个会比较麻烦，因为 Qwen 原始输出空间是图像 latent/RGB，不是 2-channel flow。更现实的做法是：
Qwen / diffusion backbone: frozen condition encoder
small DiT / U-Net: flow denoising decoder
target: 2-channel displacement field
也就是说，Qwen 只当 conditioner，不强行改它的输出头。
方案 C：Qwen 先生成粗 rectified image，再用 RAFT-style 网络预测 flow
这个方案更像“Qwen + RAFT”。
流程：
I_warp
  → Qwen Image Edit
  → rough rectified image I_qwen
然后：
(I_qwen, I_warp)
  → RAFT-style correspondence network
  → F_rect_to_warp
  → grid_sample(I_warp, F)
  → final rectified image
这个方案的好处是很直观：Qwen 提供一个大致拉直的页面作为“目标形状”，RAFT-style 网络学习从这个目标形状到原始扭曲图像的对应关系。
但是它有一个风险：如果 Qwen 生成的文字已经被改掉，那么直接用 (I_{qwen}) 和 (I_{warp}) 做 dense matching，文本内容可能对不上。因此这个方案里最好不要直接用 RGB matching，而是用：
edge map
text-line heatmap
document boundary
low-frequency layout feature
diffusion hidden feature
也就是说，Qwen 产物只提供几何布局，不参与最终文字重建。
7. 最推荐的完整训练流程
我建议你按下面四个阶段做。
Stage 1：先训练一个不含 diffusion 的 flow baseline
先不要引入 Qwen，直接训练：
I_warp → CNN / U-Net / Transformer encoder-decoder → F_hat
然后：
F_hat + I_warp → grid_sample → I_rect_hat
loss 用：
L_flow + L_img + L_smooth + L_fold
这一步的目的不是追求最终效果，而是确认你的 flow 数据、方向、grid_sample、mask 全部正确。
如果这一步都不能稳定把图片拉直，那么 diffusion 加进去也很难救。
Stage 2：加入 Qwen hidden feature，但冻结 Qwen
加载本地 Qwen-Image-Edit，使用 diffusers pipeline。官方 Hugging Face 页面给出的用法是 QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")，说明它可以在本地 pipeline 中运行。(Hugging Face)
训练时不要更新 Qwen：
with torch.no_grad():
    F_diff = qwen_feature_extractor(I_warp, prompt)
prompt 可以固定为：
Rectify the curved document page into a flat rectangular document.
Preserve the exact original text and layout.
Do not rewrite, translate, or change any characters.
但注意：这个 prompt 只是为了让 hidden feature 更关注“文档拉直”任务，不是让 Qwen 输出最终图片。
然后：
F_cnn  = CNNEncoder(I_warp)
F_diff = QwenHiddenFeature(I_warp, prompt)
F_fuse = Fusion(F_cnn, F_diff, CoordEncoding)
F_hat  = FlowDecoder(F_fuse)
训练 flow decoder 和 fusion module。
Stage 3：做高分辨率 flow refinement
文档文字很敏感，flow 不能只在低分辨率预测。建议：
coarse flow: 1/8 resolution
medium flow: 1/4 resolution
fine flow: full or 1/2 resolution
可以设计成：
Flow_1/8 → upsample → refine with 1/4 CNN feature
Flow_1/4 → upsample → refine with 1/2 edge/text feature
Flow_1/2 → convex upsample → full resolution flow
最终高分辨率 flow 再用于 grid_sample。
这里 CNN local branch 很重要，因为 Qwen/VAE latent feature 对小字笔画不一定足够精确。你的目标是 preserve text pixels，所以最后几层 refinement 一定要依赖原图的局部高分辨率特征。
Stage 4：如果需要轻量化，再做 diffusion feature distillation
Qwen-Image-Edit 很大，直接放进最终模型可能推理很慢。Qwen-Image 系列本身是 20B 级别的图像基础模型，官方 GitHub 也强调它是面向强图像生成与编辑能力的大模型。(GitHub)
所以可以训练一个 student encoder：
StudentEncoder(I_warp) ≈ QwenHiddenFeature(I_warp)
最终部署时：
I_warp → StudentEncoder + CNN Encoder → FlowDecoder → F_hat
这样训练时借用 diffusion prior，推理时不一定需要完整 Qwen。
8. 你这个任务里，最关键的技术点
我认为最关键的是这三个。
第一，flow 方向必须选对。
建议直接训练 backward flow：
[
F_{rect \rightarrow warp}
]
因为 PyTorch 的 grid_sample 本质上就是 backward sampling。这样不会出现 forward splatting 的空洞问题。
第二，diffusion 只能作为 prior，不能作为 renderer。
Qwen 可以告诉模型“纸张应该怎么被拉平”，但最后文字必须从 (I_{warp}) 采样出来。
第三，高分辨率 refinement 必须保留 CNN/local branch。
diffusion feature 负责全局弯曲趋势，CNN feature 负责字形、边缘、局部采样精度。两者不能互相替代。
9. 我建议你最终采用的 pipeline
最适合你现在数据条件的版本是：
Training data:
  I_warp, I_rect, F_gt, valid_mask

Model:
  I_warp
    ├── CNN local encoder
    ├── Frozen Qwen-Image-Edit hidden feature extractor
    ├── coordinate encoding
    ↓
  diffusion-CNN feature fusion
    ↓
  coarse-to-fine flow decoder
    ↓
  F_hat_rect_to_warp
    ↓
  grid_sample(I_warp, F_hat)
    ↓
  I_hat_rect

Loss:
  L_flow(F_hat, F_gt)
  + L_img(I_hat_rect, I_rect)
  + L_smooth(F_hat)
  + L_fold(F_hat)
一句话总结就是：
让 diffusion 学“该怎么拉直”，让 flow 负责“从原图哪里取像素”，最终图像永远由原始扭曲图像 warp 得到，而不是由 diffusion 重新生成。
这条路线能最大程度解决你担心的 VAE 信息损失和文字篡改问题，同时又能利用 Qwen Image Edit 对页面结构和几何编辑的强先验。