# V4 Layer Probing — 2026-05-15

## 这个版本做了什么

在 V3（DA-Flow 风格）的基础上，把"取 DiT 哪几层特征"变成可配置的实验参数，
系统验证 Qwen Image DiT 哪些中间层的几何先验对文档矫正最有用。

**V3 的问题：** `--num_dit_layers 4` 固定取最后 4 层（block 56~59），无法对比。  
**V4 的改进：** `--dit_target_layers "11,23,35,47"` 任意指定层，6 个实验并行跑。

## 文件结构

```
20260515-v4-layer-probe/
  train_flow_head_v4_layer_probe.py   # 训练主脚本
  eval_flow_v4_quant.py               # 定量评测脚本
  utils/
    flow_head_v4_layer_probe.py       # FlowHeadV4 模型（独立定义，不依赖 v2/v3）
  scripts/
    train_exp_A_layer12.sh            # 实验 A：单层 Layer 12（block 11）
    train_exp_B_layer24.sh            # 实验 B：单层 Layer 24（block 23）
    train_exp_C_layer36.sh            # 实验 C：单层 Layer 36（block 35）★ 推荐
    train_exp_D_layer48.sh            # 实验 D：单层 Layer 48（block 47）
    train_exp_E_layer24_36_48.sh      # 实验 E：三层融合 24+36+48 ★ 推荐
    train_exp_F_layer12_24_36_48.sh   # 实验 F：四层融合 12+24+36+48
    eval_exp_compare.sh               # 批量评测 + 打印对比表格
```

## 实验设计

| 实验 | DiT 层（0-based block） | 预期 |
|------|------------------------|------|
| A    | 11（Layer 12）         | 早层，局部纹理，全局弯曲理解弱 |
| B    | 23（Layer 24）         | 中浅层，结构开始稳定 |
| C    | 35（Layer 36）         | 中层，几何+语义最强，**主候选** |
| D    | 47（Layer 48）         | 中后层，编辑目标感知强 |
| E    | 23,35,47               | **推荐首跑**：中层多层融合 |
| F    | 11,23,35,47            | 完整多层，验证早层是否额外有益 |

## 关键设计决策

**LoRA checkpoint：** 用 `step-668000.safetensors`（纯 DiT LoRA，668k steps）。
- 不用 V3 ckpt（只含 FlowHead 权重，key 前缀是 `flow_head_v3.*`，没有 LoRA）
- 不用 V2 ckpt（step-92000，训练步数少）

**DiT 完全冻结：** 去掉 L_diffusion，只训练 FlowHead V4 + DPT Adapter。

**特征提取 timestep：** 固定从 `[400, 600)` 随机采样（中等噪声区间，特征更稳定）。

**提前 break：** 在 `max(target_layers)` 处退出 DiT 前向，不跑后续层，节省显存。

## 如何跑训练

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp

# 从推荐实验开始
bash versions/20260515-v4-layer-probe/scripts/train_exp_C_layer36.sh
bash versions/20260515-v4-layer-probe/scripts/train_exp_E_layer24_36_48.sh
```

ckpt 输出路径（各实验独立目录）：
```
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/
  20260515-exp_A_layer12/
  20260515-exp_B_layer24/
  20260515-exp_C_layer36/
  20260515-exp_D_layer48/
  20260515-exp_E_layer24_36_48/
  20260515-exp_F_layer12_24_36_48/
```

## 如何评测和对比

训练完成后，修改 `scripts/eval_exp_compare.sh` 里各实验的 `CKPT_X` 路径，然后：

```bash
# 合成验证集（有 flow_gt，算 EPE + WarpL1）
bash versions/20260515-v4-layer-probe/scripts/eval_exp_compare.sh 合成

# 真实图（silver_bullet 验证集，保存对比图）
bash versions/20260515-v4-layer-probe/scripts/eval_exp_compare.sh 真实

# 两种都跑
bash versions/20260515-v4-layer-probe/scripts/eval_exp_compare.sh both
```

真实验证集路径：
```
/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp
```
（2564 张真实弯曲文档图，无 GT flow，只做视觉对比）

对比图输出到：
```
/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/eval_vis/
  exp_C_layer36/       # 每张图：原始 | warp结果 并排
  exp_E_layer24_36_48/
  ...
```

## 与旧版本的关系

| 版本 | 训练脚本 | 特征来源 | 可回滚到 |
|------|----------|----------|----------|
| V2   | `train_flow_head.py` | 仅 CNN（两图像素输入） | 始终保留 |
| V3   | `train_flow_head_v3_daflow.py` | CNN + DiT 最后4层 Q/K | 始终保留 |
| V4   | `train_flow_head_v4_layer_probe.py` | CNN + DiT 任意指定层 Q/K | 本目录 |

所有版本代码均独立保留，互不影响。
