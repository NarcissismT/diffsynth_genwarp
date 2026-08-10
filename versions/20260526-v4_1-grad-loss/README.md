# V4.1 — Gradient Loss + InstanceNorm — 2026-05-26

## 这个版本做了什么

在 V4 的基础上做了**两个针对性修复**，目标是消除推理时的"水波纹"现象：

1. **加 gradient loss**（参考 dewarp_dino/train_warp_DDP.py 已验证可用）
   - L2 梯度损失惩罚 flow 在空间上的高频振荡
   - 每次 RAFT 迭代的 flow 都加这个 loss
   - 总损失：`L_flow + L_grad + lambda_warp * L_warp`

2. **ContextEncoder 的 BatchNorm → InstanceNorm**
   - V4 用 BN 但 ckpt 不保存 running 统计量 → 推理用初始 BN 量 → 训练-推理不一致
   - InstanceNorm 不依赖 batch 统计量，无 buffer 问题，自然解决

## 诊断证据（来自 V4 训练样本上的 flow 可视化）

V4 在 228k 步后训练样本上仍有明显问题：
- `dy.mean = -9.94`（整张图被向上推 10 像素）
- `dx/dy heatmap` 上能看到文字行轮廓 → flow 被纹理污染
- `Jacobian.min = -1.06, max=4.92, fold_ratio=0.89%` → flow 局部折叠
- 推理在真实图上 fold_ratio 上升到 5.81%

V4 的 loss 设计 `L_flow + lambda_warp * L_warp` **完全不约束 flow 的空间连续性**，
模型可以学出"文字处对、空白处乱"的 flow。这就是水波纹的根源。

## 文件结构

```
20260526-v4_1-grad-loss/
├── README.md                              # 本文档
├── train_flow_head_v4_1_grad.py           # 训练主脚本
├── qwen_image_flow_v4_1.py                # 推理主脚本（含 flow 可视化、GT sanity 模式）
├── utils/
│   └── flow_head_v4_1_grad.py             # FlowHead V4.1 + gradient_loss + sequence_loss_with_grad
└── scripts/
    ├── train_v4_1_exp_C_layer36.sh        # 实验 C：单层 Layer 36（推荐首选）
    ├── train_v4_1_exp_E_layer24_36_48.sh  # 实验 E：三层融合
    └── flow_v4_1_sample.sh                # 推理启动
```

## V4.1 vs V4 改动对比

| 项目 | V4 | V4.1 |
|------|----|------|
| ContextEncoder 归一化 | BatchNorm2d (buffer 不保存→失效) | **InstanceNorm2d**（无 buffer） |
| L_flow | 序列 L1 | 序列 L1（保持） |
| L_warp | L1(grid_sample(warped, flow), corrected) | 同（保持） |
| **L_grad** | **没有** | **L2 梯度惩罚，每次迭代都加** |
| 训练 iters | 4 | 4（保持） |
| 推理 iters | 12（不一致）/ 修后 4 | **4**（与训练一致） |
| 模型类名 | FlowHeadV4LayerProbe | FlowHeadV4_1 |
| 参数总量 | 单层 ~206 万 | 单层 ~206 万（不变，BN→IN 不影响参数数）|

## 损失函数

`utils/flow_head_v4_1_grad.py` 的 `sequence_loss_with_grad`：

```python
def gradient_loss(flow):
    """L2 梯度损失，参考 dewarp_dino"""
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]) ** 2
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2
    return (dx.mean() + dy.mean()) / 2.0


def sequence_loss_with_grad(predictions, flow_gt, gamma=0.8, gradloss_ratio=1.0):
    L_flow = sum( gamma^(N-1-i) * L1(pred_i, gt) for i in range(N) )
    L_grad = sum( gamma^(N-1-i) * gradloss_ratio * gradient_loss(pred_i) for i in range(N) )
    return L_flow, L_grad
```

训练 forward 里：
```python
L_flow, L_grad = sequence_loss_with_grad(preds, flow_gt_r, 0.8, gradloss_ratio)
total = lambda_flow * L_flow + L_grad + lambda_warp * L_warp
```

## 推荐参数

```bash
--gradloss_ratio 1.0        # 起步推荐值（dewarp_dino 默认）
--lambda_flow 1.0
--lambda_warp 0.5
--dit_target_layers "35"    # 实验 C
--learning_rate 1e-4
--num_epochs 6
--dataset_repeat 5
```

如果 1~2 万步后看到 grad_loss 不下降但 flow_loss 还在下降 → 调到 `gradloss_ratio=3.0`。
如果 fold_ratio 在 5k 步后仍 > 1% → 调到 `gradloss_ratio=5.0`。

## 训练 ckpt 输出位置

```
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_1_ckpts/
├── 20260526-v4_1-exp_C_layer36/
│   └── step-{4000,8000,...}.safetensors
└── 20260526-v4_1-exp_E_layer24_36_48/
    └── step-...
```

## 怎么跑

### 训练（从头开始，因为加了新 loss + IN 替换 BN）

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp

# 实验 C（单层 Layer 36，推荐首选）
bash versions/20260526-v4_1-grad-loss/scripts/train_v4_1_exp_C_layer36.sh

# 实验 E（三层融合）
bash versions/20260526-v4_1-grad-loss/scripts/train_v4_1_exp_E_layer24_36_48.sh
```

### 推理 + flow 诊断

修改 `scripts/flow_v4_1_sample.sh` 里 `CKPT_C` 路径为最新 step，然后：

```bash
bash versions/20260526-v4_1-grad-loss/scripts/flow_v4_1_sample.sh
```

输出会同时保存：
- `xxx_a/b/c/d.jpg`（原图、扩散结果、warp 结果、对比图）
- `xxx_flow_dx/dy/mag/quiver/jacobian/fold.png`（6 张诊断图）

每张推理图都自动带 flow 可视化，可以**直接看 fold_ratio 是不是降下来了**。

### 怎么判断 V4.1 是否成功

跑完后看 `xxx_flow_jacobian.png` 标题里的 `fold_ratio`：

| fold_ratio | 状态 |
|-----------|------|
| < 0.1% | ✓ 成功，flow 平滑 |
| 0.1% - 1% | ◯ 大幅改善，可继续训练 |
| 1% - 5% | △ 改善有限，调高 gradloss_ratio 再训 |
| > 5% | ✗ 与 V4 一样崩，需要重新检查训练日志 |

V4 真实图 fold_ratio = 5.81% → V4.1 目标 < 1%。

## 不变的部分

- 数据集（116k 样本，512×512，flow_gt 1024×1024）
- pipeline units 流水线准备 inputs
- DiT 单步前向提取 Q/K（target_layers / max_layer break 逻辑）
- LoRA 加载（step-668000.safetensors，DiT 冻结）
- ckpt 保存方式（remove_prefix_in_ckpt="flow_head."）

## 不向后兼容

V4.1 的 ckpt **不能直接用 V4 推理脚本**，因为：
- BN → IN 后 state_dict 的 key 不变（IN 也叫 norm1.weight/bias）
- 但 V4 模型有 `running_mean / running_var / num_batches_tracked` 字段，加载 V4.1 ckpt 时会缺这些 → IN 不需要它们
- 只能用 V4.1 推理脚本（导入 FlowHeadV4_1）

反向：V4 ckpt **可以加载到 V4.1**（会缺 BN 的 running 统计量，但 IN 不需要）。理论上可以做续训，但因为 V4 已经学坏了，不推荐。

## 参考

- dewarp_dino: `/juicefs-algorithm/workspace/IPT/zhuochu_yang/dewarp_dino/train_warp_DDP.py`
- diagnose.md: `versions/20260515-v4-layer-probe/diagnose.md`
- DIAGNOSE_LOG.md: `versions/20260515-v4-layer-probe/DIAGNOSE_LOG.md`
