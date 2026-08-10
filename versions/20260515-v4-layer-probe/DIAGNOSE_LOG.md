# 按 diagnose.md 五步诊断的执行记录

执行时间：2026-05-26
对应 ckpt：`flow_head_v4_ckpts/20260515-exp_C_layer36/step-236000.safetensors`

---

## Step 1：ckpt 加载验证 ✓ 已通过

修复 Bug #6 后跑了完整检查：

```
模型期待 70 key，ckpt 有 49 key
missing 总数: 21
  其中 BN buffer (允许缺失): 21
  其中关键参数缺失: 0
unexpected 总数: 0

flow_head.conv1.bias:   ✓
flow_head.conv1.weight: ✓
flow_head.conv2.bias:   ✓
flow_head.conv2.weight: ✓
```

**结论**：FlowPredHead 4 个核心参数全加载，无 unexpected key。**之前的水波纹现象不是 ckpt 加载问题导致**。Bug #6 已彻底修复。

剩 21 个缺失全是 `running_mean / running_var / num_batches_tracked` 这些 BN buffer，用初始统计量（mean=0, var=1）也能跑，但**长期看应该解决**（V5 改造）。

---

## Step 2：推理 iters 从 12 改为 4 ✓ 已实施

[qwen_image_flow_v4.py:568](qwen_image_flow_v4.py#L568) 默认值 12 → 4。
[scripts/flow_v4_sample.sh:45](scripts/flow_v4_sample.sh#L45) `FLOW_ITERS=12` → `FLOW_ITERS=4`。

**理由**：训练时 `iters=4`，GRU 只学到了 4 步内的稳定收敛。多跑 8 次容易把 flow 从"可用"推到"震荡水波纹"。RAFT 的稳定性不是来自任意增加迭代数，而是训练时对迭代过程的约束。

---

## Step 3：flow 可视化已加 ✓ 已实施

新增 `--save_flow_vis` 选项。每张推理图额外输出 6 张诊断图：

| 文件 | 含义 | 怎么看 |
|------|------|--------|
| `xxx_flow_dx.png` | dx 分量灰度图（红蓝色阶）| 应该是低频、连续；高频纹理 = flow 崩 |
| `xxx_flow_dy.png` | dy 分量 | 同上 |
| `xxx_flow_mag.png` | flow magnitude `sqrt(dx²+dy²)` | 应该和实际畸变量匹配 |
| `xxx_flow_quiver.png` | 箭头图（每 16 px 采样）| 箭头应指向同一方向 cluster；乱箭头 = 崩 |
| `xxx_flow_jacobian.png` | det(J) 热图 | 健康 flow J ≈ 1，崩坏 J 远离 1 或为负 |
| `xxx_flow_fold.png` | fold 区域（J ≤ 0 的像素）| 应该接近全黑；红色越多越糟 |

每张诊断图标题里有数值统计（mean/std/min/max/fold_ratio）。

**怎么用**：
- 如果 `_flow_dx.png` 已经是水波纹 → FlowHead 预测本身崩了，问题在模型/输入
- 如果 `_flow_dx.png` 平滑但 warp 后图像水波纹 → 问题在 grid_sample / 方向 / 尺度

---

## Step 4：GT reference sanity check ✓ 已实施

新增 `--use_gt_corrected_dir <DIR>` 选项。当提供该目录时：

```python
corrected_pil = GT (优先) or corrected_low (fallback)
flow_model(corrected_pil, warped_pil, q_feats, k_feats)
```

**用法**：
```bash
# 拷贝训练 GT 到一个目录（取一些样本）
mkdir -p /tmp/v4_sanity
cp /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/gt/Doc3d_crop/Doc3d_crop_0000000.png /tmp/v4_sanity/
cp /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/img/Doc3d_crop/Doc3d_crop_0000000.png /tmp/v4_sanity_warped/

python qwen_image_flow_v4.py \
    --ckpt_path /juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/20260515-exp_C_layer36/step-236000.safetensors \
    --dit_target_layers "35" \
    --img_size 512 \
    --input_dir /tmp/v4_sanity_warped \
    --output_dir /tmp/v4_sanity_out \
    --use_gt_corrected_dir /tmp/v4_sanity \
    --save_flow_vis
```

**判读规则**：
- A = 用 GT 做 corrected：如果**正常**矫正 → FlowHead 模型本身没问题
- B = 用 corrected_low 做 corrected：如果**水波纹** → 问题就是 corrected_low 的 domain gap，需要训练时混入 corrected_low

如果 A 也水波纹 → 问题更深，在模型结构或训练数据，不是 corrected 输入的差异。

---

## Step 5：V5 改造建议（待实施）

### 5.1 必须改的四点（diagnose.md 推荐）

```
1. 保存 BN buffer
   方案 a: 训练 forward_preprocess 时把 BN 切到 eval 模式，统计量不更新
   方案 b: 把所有 BatchNorm2d 换成 GroupNorm 或 InstanceNorm
   推荐 b，避免训练-推理 BN 模式差异

2. 训练和推理 iters 保持一致
   都用 4 或都用 6（先试 4）

3. 加 L_smooth + L_bend + L_fold（最重要）
   L_smooth = |∂u/∂x| + |∂u/∂y| + |∂v/∂x| + |∂v/∂y|
   L_bend   = |∂²u/∂x²| + |∂²u/∂y²| + |∂²v/∂x²| + |∂²v/∂y²|
   L_fold   = ReLU(eps - det(J))
   总 loss：
     L_total = L_flow + 0.5·L_warp + 0.01·L_smooth + 0.05·L_bend + 0.1·L_fold

4. 训练时混入 corrected_low（teacher forcing → diffusion forcing）
   早期：corrected_t = GT rectified
   后期：corrected_t = α·GT + (1-α)·corrected_low（α 从 1.0 线性降到 0.0）
   或随机：50% GT, 50% corrected_low
```

### 5.2 进阶建议

- 输出**低频形变场**：FlowHead 改为输出 32×32 / 64×64 coarse flow，再 bicubic upsample + smooth refinement
- 或者直接输出 TPS / B-spline 控制点，再展开成 dense flow（天然光滑）
- 训练数据增强：注入 |flow| ≈ 0 的样本（对训练集做轻微 affine 模拟近矩形输入）

### 5.3 立即可做（不重训）

- 用 Step 4 的 GT sanity check 判断问题归属
- 用 Step 3 的 flow 可视化看 flow 崩在哪里
- 跑 A/B 对比：iters=4 vs iters=12 各保存一组 c.jpg

---

## 排查执行清单

按 diagnose.md 顺序：

| 步骤 | 完成 | 备注 |
|------|------|------|
| 1. ckpt 加载检查 | ✓ | flow_head.* 全加载 |
| 2. 推理 iters 改 4 | ✓ | 默认值修改 + shell 脚本修改 |
| 3. flow 可视化 | ✓ | 加了 `--save_flow_vis` |
| 4. GT sanity check | ✓ | 加了 `--use_gt_corrected_dir` |
| 5. V5 重训 | 待做 | 见 5.1，需要修改 train_flow_head_v4 |

---

## 推荐的下一步推理命令（带所有诊断）

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp

python versions/20260515-v4-layer-probe/qwen_image_flow_v4.py \
    --ckpt_path /juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/20260515-exp_C_layer36/step-236000.safetensors \
    --dit_target_layers "35" \
    --img_size 512 \
    --flow_iters 4 \
    --save_flow_vis \
    --input_dir /juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2 \
    --output_dir /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v4_results/20260515-exp_C_layer36/vis_diagnose
```

跑完看 `_flow_dx.png / _flow_jacobian.png` 就能立刻判断 flow 是不是从源头就崩了。
