# 实验一：Qwen-Image-Edit MMDiT 零样本文档对应能力验证计划

> 版本：1.0  
> 日期：2026-08-03  
> 实验性质：独立、冻结模型、零训练、仅特征评估  
> 实验对象：`Qwen/Qwen-Image-Edit-2511`

---

## 0. 结论先行

本实验只回答一个问题：

> 在推理时只输入一张扭曲文档的条件下，Qwen-Image-Edit 的 MMDiT 在去噪过程中产生的 target Query 与 warped-source Key，是否包含可直接测量的 target-to-source 几何对应关系？

实验只进行冻结前向和离线评估：

- 不训练任何参数；
- 不接入任何下游网络；
- 不输出或使用 VAE Decoder 生成的 RGB；
- GT rectified image 和 GT backward map 只用于计算评估指标，不进入 MMDiT；
- 最终产物是不同 `layer × denoising step` 的对应能力排名、热力图、定量报告与能力判定。

实验的核心判据不是“生成图看起来是否被拉直”，而是：

\[
Q_T^{l,s}(p)^\top K_S^{l,s}(q)
\]

在源图候选位置 \(q\) 上形成的相似度分布，是否把 GT backward-map 位置

\[
q^*=M^*(p)
\]

排在前列。

---

## 1. 实验目标与假设

### 1.1 输入与特征方向

对每个样本仅输入：

```text
Warped document Iw
+ fixed rectification prompt
+ fixed random seed
→ Qwen-Image-Edit denoising trajectory
```

在第 \(s\) 个去噪步、第 \(l\) 个 MMDiT block 中提取：

\[
Q_T^{l,s}\in\mathbb{R}^{N_T\times d},
\qquad
K_S^{l,s}\in\mathbb{R}^{N_S\times d},
\]

其中：

- \(Q_T\)：target/noise token segment 的 Query；
- \(K_S\)：warped-condition token segment 的 Key；
- \(p\)：规则输出网格中的 target token；
- \(q\)：扭曲输入图中的 source token；
- 对应方向固定为 `target → warped source`，与 backward map 的定义一致。

### 1.2 原假设与备择假设

**原假设 \(H_0\)**：

MMDiT 的 target Query 与 warped-source Key 不包含有效几何对应；其匹配结果不优于随机、batch-shuffle 或恒等位置基线。

**备择假设 \(H_1\)**：

至少存在一组 `layer × denoising step × RoPE state`，使 GT source 位置稳定进入高排名候选，并在不同形变强度、文字行和页面边缘上显著优于对照组。

### 1.3 本实验最终需要给出的答案

1. 最有对应能力的是哪些 MMDiT 层？
2. 对应能力在去噪轨迹的哪个阶段出现、何时最强？
3. pre-RoPE 还是 post-RoPE Q/K 更适合几何对应？
4. 性能是否真正来自图像内容，而不是相同空间位置或 RoPE 位置偏置？
5. 结果对 target 初始噪声 seed 是否稳定？
6. 冻结 MMDiT 的对应能力应判定为“强”“条件性存在”还是“弱/不存在”？

---

## 2. 严格实验边界

### 2.1 本实验包含

- 冻结 Qwen-Image-Edit 的完整去噪轨迹；
- MMDiT 各层 target Query 与 warped-source Key 提取；
- pre-RoPE/post-RoPE 对照；
- 显式 target-to-source cost matrix；
- hard argmax、soft-argmax、top-k retrieval 与分布质量评估；
- identity、batch-shuffle 和随机候选对照；
- layer × timestep 热力图；
- 按形变强度、位移大小和结构区域进行分组统计；
- 少量多 seed 稳定性复验。

### 2.2 本实验不包含

- 任何参数训练或微调；
- 任何额外特征解码、融合或光流预测模块；
- 任何生成 RGB 的质量比较；
- 任何最终文档矫正模型的结构设计；
- 任何端到端 EPE 优化实验。

---

## 3. 与当前代码的衔接和必须先修正的问题

当前代码中可以直接复用的部分：

| 现有能力 | 文件 | 本实验用法 |
|---|---|---|
| 根据 `img_shapes` 精确切分 token segment | `docgrid_flow/providers/qwen_diffusers.py` | 继续用于分离 target 与 warped-condition tokens |
| Q/K projection hook | 同上 | 提取每层 Query/Key |
| QK Norm 与 Qwen RoPE 重放 | 同上 | 同时计算 pre-RoPE 和 post-RoPE 特征 |
| backward map、valid mask 和同步 resize | `data/dataset.py`、`data/transforms.py` | 构造 GT token correspondence |
| pixel-coordinate map 规范 | `geometry/coordinates.py` | 在 token 坐标与 source pixel 坐标间转换 |

当前实现不能直接完成本实验，原因如下：

1. `DiffusersQwenQKProbe` 强制要求 `probe_steps == 1`；
2. 多步去噪时，当前 hook 会被后续 step 覆盖，无法知道 Q/K 属于哪个 timestep；
3. 当前缓存元数据没有记录 scheduler timestep、sigma、step index、prompt、CFG、seed 和 Diffusers 版本；
4. 当前默认只抽取 `[21, 20, 18, 17]`，不能完成全层扫描；
5. 当前缓存整张 Q/K 特征，不适合大规模 `all-layer × multi-step` 实验；
6. 当前固定提示词 `Analyze the global geometry...` 不是明确的文档矫正指令，不应直接作为主实验 prompt。

### 3.1 建议新增的独立实验代码

```text
configs/mmdit_correspondence_probe.yaml
tools/build_mmdit_probe_split.py
tools/probe_mmdit_correspondence.py
tools/report_mmdit_correspondence.py
docgrid_flow/analysis/mmdit_correspondence.py
tests/test_mmdit_correspondence.py
```

不要改变现有训练阶段的缓存语义。实验探针应作为独立入口实现，避免影响当前 `format_version=3` 特征缓存。

### 3.2 新探针必须具备的能力

1. 自动读取 `len(pipe.transformer.transformer_blocks)`，不手工假设层数；
2. 记录每次 transformer forward 对应的：
   - `step_index`；
   - scheduler `timestep`；
   - `sigma` 或等价噪声强度；
   - conditional/negative forward 标记；
3. 只评估正向条件分支，避免 CFG 的 negative pass 覆盖结果；
4. 同一次 projection 同时得到 pre-RoPE 与 post-RoPE Q/K，避免重复运行模型；
5. 每层完成相似度统计后立即释放 Q/K，不把所有层、所有 timestep 的高维特征长期保存在显存或磁盘；
6. 仅保存聚合指标、每样本指标、top-k 坐标/分数和少量可视化样本；
7. 用 Diffusers 的 `callback_on_step_end` 标记去噪步边界并清理 hook 状态；
8. 所有输出带完整实验指纹：model ID、revision、Diffusers commit/version、scheduler config、prompt hash、seed、尺寸、Q/K 归一化方式和 RoPE 状态。

---

## 4. 数据集设计

### 4.1 数据来源

只从现有 validation/test 数据中抽取，不使用训练集，不进行随机增强。必须按 `document_id` 去重，防止同一文档的近似页面在 discovery 与 confirmation 子集中重复。

每个样本至少包含：

```text
warped
rectified
backward_map
valid_mask
document_id
warp_severity
hv_labels（若已有）
```

其中：

- `warped` 是 MMDiT 唯一图像输入；
- `rectified` 不进入 MMDiT，只允许用于人工核查和结果可视化；
- `backward_map` 与 `valid_mask` 只进入 evaluator；
- `hv_labels` 只用于区域分组，不进入 MMDiT。

### 4.2 三个固定子集

| 子集 | 推荐规模 | 用途 |
|---|---:|---|
| `sanity_v1` | 8 个样本 | 检查 token、坐标、方向、对照组和确定性 |
| `discovery_v1` | 64 个样本 | 扫描全部层与代表 timestep，选出候选配置 |
| `confirmation_v1` | 256 个样本 | 在独立样本上完成全 token 复验和最终排名 |

如果算力受限，最低规模可降为 `8 + 32 + 128`，但最终报告必须注明是 pilot，而不是正式结论。

### 4.3 分层抽样

优先使用现有 `warp_severity`。若该字段缺失或大量为 `unknown`，用 GT map 自动计算：

\[
d_{90}=P_{90}\left(\lVert M^*(p)-G(p)\rVert_2\right),
\]

并按 \(d_{90}\) 的四分位数构造四档形变强度。

每个子集尽量平衡：

- 轻度、中度、重度、极重度形变；
- 纯文字页、表格页、图文混排页；
- 小位移与大位移位置；
- 文字行/横纵线、页面边缘、空白背景。

---

## 5. 固定的 Qwen 推理设置

### 5.1 模型与输入尺寸

```yaml
model_id: Qwen/Qwen-Image-Edit-2511
pipeline_class: QwenImageEditPlusPipeline
height: 512
width: 512
output_type: latent
dtype: bfloat16
```

本实验固定使用当前可复现的 `Qwen-Image-Edit-2511`，中途不得切换 backbone。

### 5.2 去噪设置

主实验采用用户已经验证过的 50 步 Qwen 文档矫正轨迹：

```yaml
num_inference_steps: 50
seed: 0
```

除步数外，scheduler、CFG、negative prompt 等参数必须与现有“Qwen 单独进行 50 步文档矫正”的成功配置完全一致，并原样写入配置文件。若原配置无法恢复，则先固定一套配置，完成全部实验前不得更改。

### 5.3 固定 prompt

主实验应使用明确的文档矫正指令，而不是当前配置中的 `Analyze the global geometry...`。建议默认：

```text
Correct the geometric distortion of this document. Restore the page to a flat,
front-facing rectangle while preserving the exact text, layout, line structure,
and all visual content.
```

实验过程中不做 prompt sweep。必须保存完整 prompt 与 SHA-256 hash，确保所有样本和所有层/步使用同一提示词。

### 5.4 禁止解码 RGB

使用：

```python
output_type="latent"
```

去噪 latent 只用于产生 target Query。实验脚本不得调用 VAE Decoder，也不得根据生成 RGB 的主观效果选择 layer 或 timestep。

---

## 6. GT correspondence 的构造

设：

- target Query 网格大小为 \(h_T\times w_T\)；
- source Key 网格大小为 \(h_S\times w_S\)；
- 网络输入 source 图大小为 \(H_S\times W_S\)；
- GT backward map 使用 source pixel coordinates。

### 6.1 在 target token 中心采样 GT map

将 GT backward map 在 \(h_T\times w_T\) 的 target token 中心进行双线性采样，但不要缩放 map 数值本身，得到：

\[
M_T^*(p)=\left(x_{px}^*(p),y_{px}^*(p)\right).
\]

`valid_mask` 使用最近邻采样到 target token 网格，并与 source 边界有效性共同过滤。

### 6.2 source pixel 坐标转 Key token 坐标

在 `align_corners=False` 约定下：

\[
x_K^*=\left(x_{px}^*+\frac12\right)\frac{w_S}{W_S}-\frac12,
\]

\[
y_K^*=\left(y_{px}^*+\frac12\right)\frac{h_S}{H_S}-\frac12.
\]

反向转换为：

\[
x_{px}=\left(x_K+\frac12\right)\frac{W_S}{w_S}-\frac12,
\]

\[
y_{px}=\left(y_K+\frac12\right)\frac{H_S}{h_S}-\frac12.
\]

禁止直接用 `GT map / 16` 之类的硬编码，因为 Qwen target/source token 网格必须从实际 `img_shapes` 读取。

### 6.3 坐标单元测试

必须覆盖：

1. identity map 在任意 target/source token 分辨率下往返误差小于 `1e-4` token；
2. source resize/letterbox 后的 GT 坐标仍能正确落入 Key grid；
3. `map[...,0]` 始终为 x，`map[...,1]` 始终为 y；
4. target 与 source token 数量不相等时仍可正确计算；
5. invalid/out-of-source 点不会进入指标。

---

## 7. Cost matrix 与直接对应预测

### 7.1 特征预处理

对每个 token 进行 L2 归一化：

\[
\bar Q(p)=\frac{Q(p)}{\lVert Q(p)\rVert_2+\epsilon},
\qquad
\bar K(q)=\frac{K(q)}{\lVert K(q)\rVert_2+\epsilon}.
\]

attention heads 沿通道维拼接，但保留 Qwen 已执行的 per-head QK Norm。主实验分别计算：

1. post-QK-Norm、post-RoPE Q/K；
2. post-QK-Norm、pre-RoPE Q/K。

### 7.2 相似度矩阵

\[
C^{l,s}(p,q)=\bar Q_T^{l,s}(p)^\top \bar K_S^{l,s}(q).
\]

Discovery 阶段只对分层抽取的 target tokens 计算 \(C\)；Confirmation 阶段才对候选配置计算完整 target token 网格。

### 7.3 Hard argmax

\[
\hat q_{hard}(p)=\arg\max_q C(p,q).
\]

### 7.4 Soft-argmax

\[
P(q\mid p)=\operatorname{softmax}\left(C(p,q)/\tau\right),
\]

\[
\hat q_{soft}(p)=\sum_q P(q\mid p)q.
\]

在 Discovery 子集比较：

```text
tau ∈ {0.01, 0.03, 0.07, 0.10}
```

只允许在 Discovery 子集选择一个全局 \(\tau\)，然后冻结并用于 Confirmation，不能按测试样本单独选温度。

---

## 8. 对照组

### 8.1 Identity baseline

\[
\hat q_{id}(p)=p.
\]

用于判断低 EPE 是否只是因为多数页面形变较小。

### 8.2 Batch-shuffled Key

保持同一 target Query，但把 source Key 换成另一个文档样本：

```text
Q_target(sample i) × K_source(sample j),  i ≠ j
```

如果 post-RoPE 在 shuffle 后仍表现出很高的 identity matching，说明分数主要来自位置编码，而不是文档内容。

### 8.3 Random-candidate baseline

从有效 source token 中均匀抽取候选，用于给 top-k recall 提供理论/蒙特卡洛机会水平。

### 8.4 pre-RoPE vs post-RoPE

该对照用于判断：

- post-RoPE 是否帮助形成相对空间关系；
- 或者是否引入过强的同位置偏置。

### 8.5 Seed repeat

对 Confirmation 中固定的 32 个样本，使用：

```text
seed ∈ {0, 1, 2}
```

只复验 Discovery 选出的候选层和 timestep，不重复全层扫描。

---

## 9. 评价指标

### 9.1 第一主指标：GT 邻域 Top-k Recall

定义 GT source token 的半径 \(r\) 邻域为：

\[
\mathcal N_r(q^*)=\{q:\lVert q-q^*\rVert_\infty\le r\}.
\]

计算：

\[
R@k,r=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\mathbf 1\left[
\operatorname{TopK}_q C(p,q)\cap\mathcal N_r(q^*)\neq\varnothing
\right].
\]

固定报告：

```text
k ∈ {1, 5, 10}
r ∈ {0, 1, 2} source tokens
```

最终 layer/timestep 排名以 `R@10, r=1 token` 为第一主指标，因为它最直接回答“GT 对应是否仍保留在 cost 候选中”。

### 9.2 第二主指标：直接匹配 EPE

将 hard/soft 预测坐标转换回 source pixel coordinates，计算：

\[
EPE(p)=\lVert\hat q_{px}(p)-q^*_{px}(p)\rVert_2.
\]

报告：

- mean EPE；
- median EPE；
- EPE P95；
- hard-argmax EPE；
- soft-argmax EPE；
- 相对 identity baseline 的 EPE 降幅。

### 9.3 PCK

同时报告两套阈值：

```text
像素级：PCK@1 / PCK@3 / PCK@5 px
token级：PCK@0.5 / PCK@1 / PCK@2 source tokens
```

像素级 PCK 用于观察精细定位上限；token 级 PCK 用于避免低分辨率 MMDiT token 网格造成错误解读。

### 9.4 GT 邻域概率质量

\[
Mass_r(p)=\sum_{q\in\mathcal N_r(q^*)}P(q\mid p).
\]

同时报告：

- `GT neighborhood mass`；
- `-log(Mass_r + eps)`；
- GT 邻域最大 logit 与最强非 GT logit 的 margin。

### 9.5 Cost entropy

\[
H(p)=-\sum_qP(q\mid p)\log P(q\mid p),
\]

并除以 \(\log N_S\) 得到 `[0,1]` 归一化熵。低熵不一定代表正确，必须与 GT recall 同时判断。

### 9.6 False identity rate

只在 GT 位移大于 2 个 source tokens 的位置统计：

\[
FIR=
P\left(
\lVert\hat q(p)-p\rVert_\infty\le1
\mid
\lVert q^*(p)-p\rVert_\infty>2
\right).
\]

该指标专门检测模型是否无视形变、机械地匹配同一空间位置。

### 9.7 Seed stability

对三个 seed 报告：

- `R@10,r=1` 的均值和标准差；
- median EPE 的均值和标准差；
- top-1 source 坐标方差；
- 最佳 layer/timestep 排名一致性。

---

## 10. 分组指标

所有主指标必须分别报告：

### 10.1 按样本形变强度

```text
mild / moderate / hard / extreme
```

### 10.2 按 GT 位移

```text
0-16 px
16-48 px
48-96 px
>96 px
```

### 10.3 按结构区域

若已有 H/V/boundary 标签：

```text
text or horizontal-line region
vertical-line region
page/table boundary region
blank background region
```

若没有可靠标签，至少使用图像梯度构造 edge/non-edge 两组，但必须在报告中标记为 pseudo mask。

### 10.4 核心解释原则

- mild 样本的低 EPE 不能单独证明对应能力，因为 identity baseline 可能已经很强；
- blank background 的低熵不代表正确，因为大面积白色区域可能产生任意匹配；
- hard/extreme、大位移、文字行和边缘区域是最关键的判断依据；
- 若 soft-argmax EPE 较差但 `R@10,r=1` 很高，说明 cost 是多峰的，不能直接判定特征没有对应信息。

---

## 11. 具体执行流程

### Step 0：冻结实验配置

建立 `configs/mmdit_correspondence_probe.yaml`，固定：

- model ID/revision；
- Diffusers version/commit；
- scheduler 全部参数；
- 50 步 timestep/sigma 列表；
- prompt 与 negative prompt；
- CFG 参数；
- 输入尺寸；
- seed；
- target/source token segment index；
- Q/K Norm 和 RoPE 位置；
- Discovery 与 Confirmation 样本 ID。

配置冻结后，中途不得根据结果修改 prompt 或数据子集。

### Step 1：实现多步、逐层在线探针

完成以下功能：

1. 识别当前 denoising step；
2. 提取所有 MMDiT block 的 target Query 和 source Key；
3. 区分 conditional 与 negative forward；
4. 在选定 timestep 在线计算匹配指标；
5. 每层评分完成后立即释放特征；
6. 支持只计算部分 target tokens 或完整 target lattice；
7. 输出 JSONL/Parquet/CSV 指标和少量 top-k 结果。

### Step 2：Sanity Gate

在 `sanity_v1` 上完成：

- token segment 长度与 `img_shapes` 完全一致；
- Query 来自 target segment，Key 来自 warped condition segment；
- GT pixel↔token 坐标往返测试通过；
- 相同配置、相同 seed 重复两次结果一致；
- batch-shuffled Key 的结果明显劣化；
- identity baseline 计算正确；
- 无 NaN、Inf、越界 top-k 或无效点泄漏。

Sanity Gate 未通过时，不允许进入正式扫描。

### Step 3：Discovery 全层扫描

数据：`discovery_v1`，64 个样本。

MMDiT：运行完整 50 步轨迹，但只在以下代表 step 评分：

```text
step_index = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
```

同时保存实际 scheduler timestep/sigma，最终比较以噪声强度为准，而不是只看 step index。

层：扫描全部 MMDiT blocks；当前 `[17,18,20,21]` 只作为已用配置锚点，不享受优先选择。

空间采样：每个样本抽取 128 个有效 target tokens，建议比例：

```text
50%：文字行/横纵线/高梯度区域
25%：页面或表格边缘
25%：空白/低纹理背景
```

形变位移各档保持平衡。对每个 layer/step 同时评估 pre-RoPE 与 post-RoPE。

输出三张主热力图：

1. `R@10,r=1` 的 layer × step 热力图；
2. median soft-argmax EPE 的 layer × step 热力图；
3. false identity rate 的 layer × step 热力图。

候选选择规则：

1. 先按 `R@10,r=1` 排序；
2. 再用 hard/soft EPE、hard subset recall 和 false identity rate 打破并列；
3. 保留最多 6 个单层配置；
4. 候选至少覆盖两个不同深度区间，避免 64 个 Discovery 样本导致单点偶然最优。

### Step 4：Confirmation 独立复验

数据：`confirmation_v1`，256 个未参与选择的样本。

只评估 Discovery 选出的最多 6 个 `layer × step × RoPE` 配置；每个配置使用完整有效 target token 网格。

同时保留：

- identity baseline；
- batch-shuffled Key；
- random baseline；
- 当前 `[17,18,20,21]` 配置中表现最好的锚点。

只用 Confirmation 结果生成最终排名，Discovery 结果不能作为最终指标。

### Step 5：Seed 稳定性复验

从 Confirmation 固定抽取 32 个样本，只对最终前 3 个配置运行 seeds `{0,1,2}`。

不重新选择 layer/timestep，只验证排名和对应坐标是否稳定。

### Step 6：生成最终实验报告

报告必须包含：

1. 完整实验配置与代码版本；
2. Discovery layer × step 热力图；
3. Confirmation 总体和分组指标；
4. pre/post-RoPE 对比；
5. identity、shuffle、random 对照；
6. seed 稳定性；
7. 至少 24 个定性样本；
8. 失败案例分类；
9. 对 MMDiT 对应能力的单一结论：强、条件性存在或弱/不存在。

---

## 12. 可视化要求

### 12.1 Layer × step 热力图

每个格子是一组 MMDiT `layer × denoising step`，至少绘制：

- `R@10,r=1`；
- median EPE；
- normalized entropy；
- false identity rate。

pre-RoPE 与 post-RoPE 分开作图，色标范围保持一致。

### 12.2 单点 cost map

对选定 target 点 \(p\)，在 warped source token 网格上显示：

- 完整 cost heatmap；
- GT source 坐标；
- top-1；
- top-5/top-10 候选；
- identity 位置；
- GT 邻域半径。

### 12.3 稀疏对应连线图

每个样本选择 16-32 个 target 点，分别绘制 GT 和预测 source 位置。必须覆盖：

- 成功样本；
- 重复文字/重复表格线；
- 大面积白色背景；
- 强弯曲页面边缘；
- 大位移区域；
- post-RoPE identity-bias 失败样本。

### 12.4 分组分布图

对不同形变等级和结构区域绘制 EPE/Recall 箱线图或误差条，不能只展示总体平均值。

---

## 13. 预注册能力判定标准

最终以 Confirmation 集为准，且 `R@10,r=1` 权重高于 soft-argmax EPE。

### 13.1 强对应能力

同时满足：

1. 总体 `R@10,r=1 ≥ 60%`；
2. hard/extreme 子集 `R@10,r=1 ≥ 40%`；
3. 文字/边缘区域 `R@10,r=1 ≥ 45%`；
4. 至少为 shuffled/random 对照的 5 倍；
5. soft-argmax median EPE 相对 identity baseline 下降至少 20%；
6. 大位移位置的 false identity rate `< 30%`；
7. 三个 seed 的 `R@10,r=1` 标准差不超过 5 个百分点。

### 13.2 条件性对应能力

出现下列任一情况：

- 总体 `R@10,r=1` 位于 30%-60%；
- 总体较强，但只在 mild/small-displacement 上成立；
- top-k recall 较高，但 soft-argmax 因多峰分布而误差较大；
- pre-RoPE 明显有效，而 post-RoPE 被 identity bias 主导；
- 最佳配置对 seed 或 timestep 较敏感；
- 文字/边缘有效，但空白/重复纹理明显失败，或反之。

报告必须准确指出能力成立的层、步、区域和形变范围，不能简单写“有效”。

### 13.3 弱或无对应能力

满足以下整体模式：

1. 最佳配置 `R@10,r=1 < 30%`；
2. 不超过 shuffled/random 对照的 2 倍；
3. hard/extreme 或大位移区域接近机会水平；
4. direct EPE 不优于 identity baseline；
5. post-RoPE 的低误差主要由同位置偏置造成；
6. 最佳层/步在不同 seed 下不能复现。

阈值应在正式运行前写入配置和报告模板，不能根据最终结果事后修改。

---

## 14. 输出目录与文件规范

```text
runs/mmdit_correspondence_exp1/
├── frozen_config.yaml
├── environment.json
├── splits/
│   ├── sanity_v1.jsonl
│   ├── discovery_v1.jsonl
│   └── confirmation_v1.jsonl
├── discovery/
│   ├── aggregate_metrics.json
│   ├── per_sample_metrics.parquet
│   └── heatmaps/
├── confirmation/
│   ├── aggregate_metrics.json
│   ├── per_sample_metrics.parquet
│   └── subgroup_metrics.json
├── seed_stability/
├── visualizations/
├── selected_configs.json
└── MMDiT_correspondence_experiment_1_report.md
```

禁止为全部样本保存全部层、全部 timestep 的原始 3072 维 Q/K；这会产生极大的存储开销且并非本实验所需。仅对少量定性案例保存 cost map 和 top-k 候选。

---

## 15. 资源预算与执行顺序

### 15.1 推荐执行顺序

| 阶段 | 样本数 | Qwen 轨迹 | 主要计算 |
|---|---:|---:|---|
| Sanity | 8 | 50 steps | 少量层/点，检查正确性 |
| Discovery | 64 | 50 steps | 全层、11 个代表 step、每图 128 target tokens |
| Confirmation | 256 | 50 steps | 最多 6 个候选、完整 target lattice |
| Seed repeat | 32 × 3 seeds | 50 steps | 最终前 3 配置 |

总 wall time 应先用 8 个 sanity 样本实测：

\[
T_{total}\approx
T_{50step/image}\times(8+64+256+32\times3)
+T_{cost}.
\]

不要在没有 pilot 计时的情况下直接估算小时数。多 GPU 时按样本并行，每张卡独立加载一份模型并写入独立 shard，最后只合并指标文件。

### 15.2 显存与存储策略

- `batch_size=1`；
- BF16 运行 MMDiT；
- cost 计算可用 BF16 matmul，指标累积转 FP32；
- Discovery 每图最多 128 个 target tokens；
- source tokens 可分块计算 top-k 和 log-sum-exp；
- 每层评分后立即释放 Q/K；
- 不保存完整 raw feature tensor；
- 每个 worker 写独立 JSONL/Parquet shard，防止并发覆盖。

---

## 16. 完成定义

实验一只有在以下内容全部完成时才算结束：

- [ ] 三个数据子集固定并保存 sample IDs；
- [ ] 坐标、token segment、Q/K 方向的单元测试全部通过；
- [ ] 50 步去噪中的 layer/step 标记准确；
- [ ] Discovery 完成全部层扫描；
- [ ] Confirmation 使用独立数据完成复验；
- [ ] identity、shuffle、random、pre/post-RoPE 对照齐全；
- [ ] seed 稳定性完成；
- [ ] 总体、形变、位移和结构分组指标齐全；
- [ ] 热力图与 24 个以上定性案例生成；
- [ ] 最终结论严格归为“强”“条件性存在”或“弱/不存在”；
- [ ] 报告中没有使用生成 RGB 的主观质量作为判断依据。

---

## 17. 依据

DA-Flow 的零样本特征分析同样在不训练光流网络的条件下，从扩散模型注意力层抽取 Query/Key，构建全对全 dot-product cost volume，通过 soft-argmax 得到直接对应，并比较不同层和去噪 timestep 的 EPE。本实验保留这一验证逻辑，但把帧间对应改为单图编辑中的 `target token → warped-condition token` 对应，并以真实 GT backward map 作为监督参考。

- [DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models](https://arxiv.org/abs/2603.23499)，重点见第 4.3 节、式（11）和式（12）；本项目所用本地版本为 `project_sources/01-DA-Flow.pdf`。
- [Qwen-Image 官方仓库](https://github.com/QwenLM/Qwen-Image)：记录 Qwen-Image-Edit-2511 的官方模型与调用方式。
- [Diffusers Qwen-Image-Edit Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py)：支持传入 `latents`、`output_type="latent"` 和 `callback_on_step_end`，可用于多步轨迹探针而不调用最终 VAE Decoder。

