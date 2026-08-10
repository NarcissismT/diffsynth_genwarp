我判断：**图 3 的“水波纹”不是 diffusion 50 步生成图的问题，而是 FlowHead 预测出的 backward flow 已经崩了，随后 `grid_sample` 把原图按错误位移场重采样，导致文字和插图被局部来回拉扯。** 你的设计本来是让 Qwen 先产生低清矫正图 `corrected_low`，再让 FlowHead 输出 backward flow，用这个 flow 去原始 warped 图中采样，从而保留原始像素文字 。所以只要 flow 本身出现高频震荡，最终图像就会呈现这种水波纹。

### 最可疑原因：FlowPredHead 权重没有正确加载

你上传的 `PIPELINE.md` 里其实已经指出了一个和当前现象高度一致的问题：之前推理时多做了 `sd.replace("flow_head.", "", 1)`，会把内层 `FlowPredHead` 的 `flow_head.conv1.weight` 错误剥成 `conv1.weight`，导致输出层加载失败，最后 `FlowPredHead` 使用随机权重；文档也明确写到这会让 GRU hidden 经过随机 conv 输出随机 Δflow，并且 12 次迭代累积成大幅错乱位移，形成“水波纹崩坏” 。

你现在这张图 3 的形态非常像这个 bug：不是整体轻微错位，而是**局部高频、连续、类似波纹的扭曲**。这通常不是 diffusion 输出造成的，而是 flow 的 `dx/dy` 在空间上剧烈振荡。

你第一步应该直接检查：

```python
sd = load_file(ckpt_path)
missing, unexpected = flow_model.load_state_dict(sd, strict=False)

print("missing:", missing)
print("unexpected:", unexpected)

bad_missing = [
    k for k in missing
    if not (
        "running_mean" in k
        or "running_var" in k
        or "num_batches_tracked" in k
    )
]

assert len(unexpected) == 0, unexpected
assert len(bad_missing) == 0, bad_missing
assert not any(k.startswith("flow_head.") for k in bad_missing)
```

重点看这些 key 有没有缺失：

```text
flow_head.conv1.weight
flow_head.conv1.bias
flow_head.conv2.weight
flow_head.conv2.bias
```

只允许缺 BN buffer；如果 FlowPredHead 相关参数缺失，图 3 崩坏基本就能解释了。

---

### 第二个高风险点：推理迭代 12 次，但训练只有 4 次

你的训练 forward 里 FlowHead 是 `iters=4`，而推理调用是 `iters=12` 。这对 RAFT 风格网络很危险。RAFT 的核心是构建相关性体，然后通过 recurrent unit 迭代更新单个 flow field；它的稳定性来自训练时对迭代更新过程的约束，而不是任意增加迭代次数 ([ECVA][1])。如果你的 GRU update 没学会 12 步稳定收敛，多跑 8 步很容易把 flow 从“可用”继续推到“过矫正/震荡”。

建议立刻把推理改成：

```python
flow_low = flow_model(
    corrected_t,
    warped_t,
    q_features,
    k_features,
    iters=4,
)
```

然后分别保存第 1、2、3、4、8、12 次的 warp 结果。若 4 次还可以、12 次变成水波纹，说明主要是 **over-refinement**。

---

### 第三个问题：训练用 GT rectified，推理用 corrected_low，存在输入分布差异

训练时 FlowHead 的 corrected 分支吃的是 **GT rectified**，但推理时只能吃 Qwen 生成的 `corrected_low`，你的文档中也把这一项标成训练-推理不一致 。这会带来一个很大的 domain gap：

训练时：

```text
corrected_t = GT rectified
warped_t    = warped
```

推理时：

```text
corrected_t = Qwen corrected_low
warped_t    = warped
```

图 2 虽然几何大体拉直了，但文字内容已经被 diffusion / VAE 改写或模糊。FlowHead 如果依赖 `corrected_low` 的局部文字纹理做相关性匹配，就可能把错误纹理当成几何信号，进而产生局部震荡 flow。

建议做一个非常关键的 ablation：

```text
A. 推理时 corrected_t = GT rectified       # 在验证集上可做
B. 推理时 corrected_t = Qwen corrected_low
```

如果 A 正常、B 水波纹，说明问题不是 flow head 结构本身，而是 **corrected_low 作为 reference 太不稳定**。后续训练应加入 teacher forcing / scheduled reference：

```text
训练前期：corrected_t = GT
训练中后期：corrected_t = α * GT + (1 - α) * corrected_low
或随机使用：
50% GT rectified
50% Qwen corrected_low
```

---

### 第四个问题：没有 smoothness / fold 正则，dense flow 太自由

你的 V4 loss 目前主要是 flow L1 和 warp L1，而且文档明确写了没有 smoothness loss，也没有 fold / Jacobian regularization 。对于文档矫正任务，这会很危险，因为真实页面形变通常是**低频、连续、局部平滑**的，而不是像光流那样允许每个像素自由移动。

图 3 的水波纹本质就是：模型预测了一个空间高频的 dense displacement field。

建议加入：

```text
L_total =
  L_flow
+ λ_warp L_warp
+ λ_smooth L_smooth
+ λ_bend L_second_order
+ λ_fold L_jacobian
+ λ_identity L_identity
```

其中最重要的是：

```python
# 一阶平滑
L_smooth = |∂u/∂x| + |∂u/∂y| + |∂v/∂x| + |∂v/∂y|

# 二阶弯曲能量，更适合文档 dewarp
L_bend = |∂²u/∂x²| + |∂²u/∂y²| + |∂²v/∂x²| + |∂²v/∂y²|

# 防止 fold-over
J = det([[1 + ∂u/∂x, ∂u/∂y],
         [∂v/∂x, 1 + ∂v/∂y]])
L_fold = ReLU(eps - J)
```

文档矫正里我更建议你不要直接输出完全自由的 per-pixel flow，而是输出一个**低频形变场**，例如：

```text
coarse flow 32×32 / 64×64
→ bicubic upsample
→ smooth residual refinement
```

或者直接输出 TPS / B-spline control points，再展开成 dense flow。这样天然能减少水波纹。

---

### 第五个必须检查：backward flow 的方向、尺度、归一化

你的 flow 语义是 backward flow：`flow[y,x]=(dx,dy)`，表示矫正图上的 `(x,y)` 应该去弯曲图 `(x+dx, y+dy)` 处采样 。所以最终 warp 时应该是：

```text
target rectified coordinate: (x, y)
source warped coordinate:   (x + dx, y + dy)
```

也就是：

```python
sample_x = base_x + flow_x
sample_y = base_y + flow_y
```

然后再转成 `grid_sample` 需要的 normalized coordinate：

```python
grid_x = 2 * sample_x / (W - 1) - 1
grid_y = 2 * sample_y / (H - 1) - 1
```

需要同时检查四件事：

```text
1. flow 是 backward 还是 forward？
2. dx/dy 有没有写反？
3. x/y 有没有写反？
4. flow 从 512 放大到原图时，dx 是否乘 W_orig / W_low，dy 是否乘 H_orig / H_low？
```

如果方向错，通常会整体拉歪；如果尺度错，会严重撕裂；如果 dx/dy 交换，会出现横纵方向异常扭曲；如果输出 head 随机或迭代过多，就更像你现在这种水波纹。

---

### 我建议你按这个顺序排查

**第一步：只验证 ckpt 加载。**
确保除了 BN buffer 以外没有任何 missing key，尤其是 `flow_head.*` 不能缺。

**第二步：把推理 iters 从 12 改成 4。**
训练 4 次，推理也先 4 次。先不要追求更强 refine。

**第三步：可视化 flow，而不是只看 warp 图。**

保存：

```text
dx heatmap
dy heatmap
flow magnitude
flow quiver
Jacobian determinant
fold ratio
```

如果 `dx/dy` 上已经有密集水波纹，说明 FlowHead 预测本身崩了；如果 `dx/dy` 平滑但 warp 后水波纹，说明是 `grid_sample / flow direction / scaling` 的问题。

**第四步：做 GT reference sanity check。**

在有 GT 的验证集上跑：

```text
flow_model(GT_rectified, warped, q, k)
```

如果这时正常，而：

```text
flow_model(Qwen_corrected_low, warped, q, k)
```

崩坏，那就是 corrected_low domain gap。

**第五步：重新训练一版更稳定的 V5。**

我建议 V5 至少改四点：

```text
1. 保存 BN buffer，或者把 BatchNorm 全部换成 GroupNorm / InstanceNorm
2. 训练和推理 iters 保持一致，例如都用 4 或都用 6
3. 加 L_smooth + L_bend + L_fold
4. 训练时混入 corrected_low，减少 GT reference 到 diffusion reference 的分布差异
```

---

### 和 FlowDiffuser / RAFT 思路的关系

你现在的方法更像是“冻结 diffusion model，用 DiT 中间层 Q/K 作为几何先验，再接 RAFT-style recurrent flow head”。这和 FlowDiffuser 还不完全一样。FlowDiffuser 的核心是直接把 optical flow 当作 diffusion 生成对象，通过 reverse denoising 从噪声逐步生成 flow，而不是先生成图像再后接一个 flow head ([CVF Open Access][2])。你的 FlowHead 则更接近 RAFT 的相关性体 + recurrent update 范式 ([ECVA][1])。因此当前最优先不是继续调 diffusion 步数，而是先让 **flow field 本身稳定、平滑、方向正确、训练推理一致**。

我建议你下一步先跑：**ckpt key 检查 + iters=4 + flow 可视化**。这三项基本能定位 70% 以上的问题。

