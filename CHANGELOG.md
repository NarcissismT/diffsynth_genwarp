# Changelog

---

## V4 Layer Probing（2026-05-15）

### 背景

在 V3（DA-Flow 风格）中，`--num_dit_layers N` 固定取 Qwen Image DiT 的**最后 N 层**
（即 block 60-N ~ block 59）。这使得无法系统验证"哪些中间层的几何先验最强"。

V4 的目标是把这个问题变成一个**可验证的 layer selection probing 实验**：
通过 `--dit_target_layers "11,23,35,47"` 任意指定层，用验证集 flow EPE
和 warp reconstruction loss 排序，确定最优层组合。

### 新增文件（不修改任何旧文件）

| 文件 | 说明 |
|------|------|
| [utils/flow_head_v4_layer_probe.py](utils/flow_head_v4_layer_probe.py) | FlowHeadV4 模型定义，所有基础模块独立定义不依赖 v2/v3 |
| [train_flow_head_v4_layer_probe.py](train_flow_head_v4_layer_probe.py) | 训练主脚本，新增 `--dit_target_layers` 参数 |
| [eval_flow_v4_quant.py](eval_flow_v4_quant.py) | 定量评测脚本（合成数据 EPE + 真实图对比图）|
| [scripts/train_exp_A_layer12.sh](scripts/train_exp_A_layer12.sh) | 实验 A：单层 Layer 12（block 11）|
| [scripts/train_exp_B_layer24.sh](scripts/train_exp_B_layer24.sh) | 实验 B：单层 Layer 24（block 23）|
| [scripts/train_exp_C_layer36.sh](scripts/train_exp_C_layer36.sh) | 实验 C：单层 Layer 36（block 35）★ 推荐首跑 |
| [scripts/train_exp_D_layer48.sh](scripts/train_exp_D_layer48.sh) | 实验 D：单层 Layer 48（block 47）|
| [scripts/train_exp_E_layer24_36_48.sh](scripts/train_exp_E_layer24_36_48.sh) | 实验 E：三层融合 Layer 24+36+48 ★ 推荐首跑 |
| [scripts/train_exp_F_layer12_24_36_48.sh](scripts/train_exp_F_layer12_24_36_48.sh) | 实验 F：四层融合 Layer 12+24+36+48 |
| [scripts/eval_exp_compare.sh](scripts/eval_exp_compare.sh) | 批量评测 + 打印对比表格 |

旧文件保持不变（可随时回滚）：
- `train_flow_head.py` / `train_flow_head_v2.sh` — V2：LoRA + FlowHeadV2（双图像素输入）
- `train_flow_head_v3_daflow.py` / `train_flow_head_v3_daflow.sh` — V3：取最后 N 层 Q/K

### V4 vs V3 核心差异

| 项目 | V3 | V4 |
|------|----|----|
| DiT 层选择 | 固定取最后 `num_dit_layers` 层（硬编码） | 任意指定，`--dit_target_layers "23,35,47"` |
| L_diffusion | 存在（与 LoRA 联合训练时计算） | 去掉，DiT 完全冻结，只计算 L_flow + L_warp |
| DiT 提前退出 | 在 `max(target_indices)` 处 break | 同，节省后续层的显存和时间 |
| 特征提取 timestep | 随机均匀采样 `[0, 1000)` | 固定采样 `[400, 600)`（中等噪声，σ ≈ 0.4~0.6） |
| lambda_warp 默认值 | 0.1 | 0.5（增大 warp 重建约束） |
| LoRA 来源 | step-92000（V2 联合训练） | step-668000（纯 DiT LoRA，668k steps 更充分）|

### LORA_CKPT 选择说明

V3 ckpt（`flow_head_v3_daflow_ckpts/step-*.safetensors`）**只含 FlowHead 权重**，
key 全是 `flow_head_v3.*`，没有 LoRA。因此：

- `--lora_checkpoint`（DiT 特征质量）→ 使用 `step-668000.safetensors`（纯 DiT LoRA）
- `--flow_head_init`（FlowHead 热启动）→ 若需要可用 V3 ckpt 去前缀转换（见 `utils/convert_v3_ckpt_to_v4_init.py`，默认不使用）

### 评测方式

**合成数据（有 flow_gt）**：

```bash
# 单个实验
python eval_flow_v4_quant.py \
    --ckpt_path /path/to/step-XXXXX.safetensors \
    --dit_target_layers "35" \
    --exp_name "exp_C_layer36"
```

**真实图（silver_bullet 验证集，无 flow_gt）**：

```bash
python eval_flow_v4_quant.py \
    --ckpt_path /path/to/step-XXXXX.safetensors \
    --dit_target_layers "35" \
    --real_val_dir /juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp \
    --vis_output_dir /path/to/vis_output \
    --exp_name "exp_C_layer36"
```

**批量对比（6个实验跑完后）**：

```bash
# 先在 eval_exp_compare.sh 中填好各实验的 checkpoint 路径，然后：
bash scripts/eval_exp_compare.sh 合成   # 只跑合成数据评测
bash scripts/eval_exp_compare.sh 真实   # 只跑真实图推理 + 对比图
bash scripts/eval_exp_compare.sh both  # 两种都跑
```

脚本会自动打印排序后的对比表格：

```
[合成数据（有 flow_gt）]
================================================================
实验名                           层配置       样本数       EPE ↓   WarpL1 ↓
----------------------------------------------------------------
exp_E_layer24_36_48              23,35,47       600      0.8xxx    0.0xxx
exp_C_layer36                    35             600      0.9xxx    0.0xxx
...
================================================================
```

### 实验设计说明

```
实验 A：Layer 12   → 早层，局部纹理/边缘
实验 B：Layer 24   → 中浅层，结构开始稳定
实验 C：Layer 36   → 中层，几何+语义最强（主要候选）
实验 D：Layer 48   → 中后层，编辑目标感知强
实验 E：24+36+48   → 推荐第一版多层组合
实验 F：12+24+36+48 → 完整多层，验证早层是否额外有益
```

评估指标：
- 验证集 flow EPE（End-Point Error）：`mean(|F_hat - F_gt|_2)`
- 验证集 warp L1：`mean(|grid_sample(I_warp, F_hat) - I_rect|_1)`

ckpt 输出路径：
```
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/
  20260515-exp_A_layer12/
  20260515-exp_B_layer24/
  20260515-exp_C_layer36/
  20260515-exp_D_layer48/
  20260515-exp_E_layer24_36_48/
  20260515-exp_F_layer12_24_36_48/
```

---

## V3 DA-Flow 风格（2026-05-13）

### 新增文件

| 文件 | 说明 |
|------|------|
| `utils/flow_head_v3_daflow.py` | FlowHeadV3DAFlow：CNN + DiT 最后 N 层 Q/K 融合 |
| `utils/dpt_head.py` | DPT 上采样头，将 DiT token 特征上采样到 H/8 |
| `train_flow_head_v3_daflow.py` | 训练主脚本，支持 `--freeze_dit` 和 `--num_dit_layers` |
| `scripts/train_flow_head_v3_daflow.sh` | 训练脚本，加载 V2 LoRA ckpt，冻结 DiT |

### 设计

- Stage 2 模式：加载已有 LoRA ckpt 作为 DiT 权重，冻结后只训练 DPT Head + FlowHead V3
- 提取 DiT 最后 `num_dit_layers=4` 层的 img Q/K 特征
- 在 `max(target_indices)` 处提前 break，不跑后续层

---

## V2 FlowHead（2026-05-12）

### 新增文件

| 文件 | 说明 |
|------|------|
| `utils/flow_head_v2.py` | FlowHeadV2：轻量 RAFT 风格，双图像素输入，~60万参数 |
| `train_flow_head.py` | 联合训练：DiT LoRA + FlowHeadV2，L = L_diff + L_flow + L_warp |
| `scripts/train_flow_head.sh` | 训练启动脚本 |

### 设计

- FlowHead V2 输入：`(corrected_gt, warped)` 两张像素图
- ContextEncoder 初始化 GRU hidden state，CorrBlock 计算 all-pairs 相似度
- 训练时 LoRA 可训练，FlowHead 可训练，其余冻结
