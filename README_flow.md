# 文档矫正系统：Diffusion + Optical Flow

## 目录

- [问题背景](#问题背景)
- [整体架构](#整体架构)
- [模型详细流程](#模型详细流程)
  - [Stage 1：扩散模型推理](#stage-1扩散模型推理)
  - [Stage 2：光流估计（V2）](#stage-2光流估计v2)
  - [Stage 2：光流估计（V3 DA-Flow）](#stage-2光流估计v3-da-flow)
  - [Stage 3：高清 Warp](#stage-3高清-warp)
- [训练方案](#训练方案)
- [推理方案对比](#推理方案对比)
- [版本演进](#版本演进)
- [快速开始](#快速开始)
- [文件结构](#文件结构)

---

## 问题背景

文档矫正任务存在一个核心矛盾：

```
直接用扩散模型矫正：
  warped 图 → DiT 推理 → corrected 图
  ✓ 几何矫正正确
  ✗ 文字模糊、偶有错字（VAE 编解码损失高频细节）

直接用光流矫正（RAFT）：
  warped 图 + corrected 图 → RAFT → 光流 → warp
  ✓ 像素来自原图，文字清晰
  ✗ RAFT 在退化/折痕区域精度不够（缺乏几何语义理解）
```

**本系统的解决方案**：让扩散模型负责"理解几何"，让光流网络负责"精确像素匹配"，最终像素从原始高清图采样，绕过 VAE 损失。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           推理时数据流                                   │
│                                                                         │
│  原始高清 warped 图（如 2964×2084）                                      │
│         │                                                               │
│         ├──[resize to 1024]──────────────────────────────────┐          │
│         │                                                    │          │
│         ▼                                                    │ (保留原图) │
│  ┌─────────────────────────────────────────┐                │          │
│  │         QwenImage DiT + LoRA            │                │          │
│  │                                         │                │          │
│  │  warped_1024 作为 condition (edit_image) │                │          │
│  │  noise ──→ 50步去噪 ──→ clean latent    │                │          │
│  │                ↓                        │                │          │
│  │         VAE decode                      │                │          │
│  │                ↓                        │                │          │
│  │    corrected_low（1024×1024）← b_*.jpg  │                │          │
│  │    [V3] Q/K 特征（top-4层注意力）       │                │          │
│  └─────────────────────────────────────────┘                │          │
│         │ corrected_low + [V3: Q/K feats]                   │          │
│         │ + warped_1024                                      │          │
│         ▼                                                    │          │
│  ┌─────────────────────────────────────────┐                │          │
│  │        FlowHead（V2 或 V3）             │                │          │
│  │                                         │                │          │
│  │  输出：光流 flow（1, 2, 1024, 1024）    │                │          │
│  │  方向：corrected 每个像素               │                │          │
│  │        → 应去 warped 的哪个位置采样     │                │          │
│  └─────────────────────────────────────────┘                │          │
│         │ flow_1024                                          │          │
│         ▼                                                    │          │
│  upscale_flow（双线性插值 + 偏移量缩放）                     │          │
│         │ flow_hires（1, 2, 2964, 2084）                     │          │
│         ▼                                                    │          │
│  grid_sample(warped_hires, flow_hires) ←──────────────────┘          │
│         │                                                               │
│         ▼                                                               │
│  最终高清矫正结果（2964×2084）← c_*.jpg                                 │
│  ✓ 几何正确（来自扩散模型的矫正理解）                                    │
│  ✓ 文字清晰（像素直接来自原始高清图）                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 模型详细流程

### Stage 1：扩散模型推理

```
输入：warped_1024（1, 3, 1024, 1024），值域 [0, 255]
      prompt（文本描述矫正任务）

┌──────────────────────────────────────────────────────────────────┐
│  QwenImage DiT（60层 Transformer，hidden_size=3072）             │
│                                                                  │
│  Step 1: VAE encode warped_1024                                  │
│          warped_1024 (1,3,1024,1024)                             │
│          → warped_latent (1,16,128,128)  ← 条件 latent           │
│                                                                  │
│  Step 2: 初始化噪声                                              │
│          noise ~ N(0,I)  shape: (1,16,128,128)                  │
│                                                                  │
│  Step 3: 50步 flow matching 去噪                                 │
│          t=1.0 (纯噪声) ──→ t=0.0 (干净图像)                    │
│          每步：noise_pred = DiT(z_t, t, warped_latent, prompt)   │
│                z_{t-1} = scheduler.step(z_t, noise_pred)        │
│                                                                  │
│          [V3] 在去噪过程中 hook top-4 层的 Q/K 注意力特征        │
│          对所有 timestep 的特征取均值，得到稳定的几何特征        │
│          Q/K feats: list[4] of (1, T, 3072)，T=(128×128)=16384  │
│                                                                  │
│  Step 4: VAE decode 最终 latent                                  │
│          clean_latent (1,16,128,128)                             │
│          → corrected_low (1,3,1024,1024)  ← b_*.jpg             │
└──────────────────────────────────────────────────────────────────┘

输出：corrected_low（矫正后像素图，文字略糊）
      [V3] q_features, k_features（各4个，每个 (1,16384,3072)）
```

---

### Stage 2：光流估计（V2）

**FlowHead V2**（~137万参数，纯像素输入，参考 RAFT）

```
输入：corrected_low (1,3,1024,1024)，值域 [-1,1]
      warped_1024   (1,3,1024,1024)，值域 [-1,1]

┌──────────────────────────────────────────────────────────────────┐
│                        FlowHead V2                               │
│                                                                  │
│  Step 1: Feature Encoder（corrected 和 warped 共用权重）         │
│          输入 (1,3,1024,1024)                                    │
│          → Conv(3→32, 7×7, stride=2)  → (1,32,512,512)         │
│          → ResBlock(32→64, stride=2)  → (1,64,256,256)         │
│          → ResBlock(64→96, stride=2)  → (1,96,128,128)         │
│                                                                  │
│          F_c = feature_encoder(corrected)  (1,96,128,128)       │
│          F_w = feature_encoder(warped)     (1,96,128,128)       │
│                                                                  │
│  Step 2: Context Encoder（只处理 corrected）                     │
│          输入 corrected (1,3,1024,1024)                          │
│          → 多层卷积 → (1,128,128,128)                           │
│          → 分裂：hidden (1,96,128,128) + context (1,32,128,128) │
│                                                                  │
│  Step 3: Correlation Volume（all-pairs 相似度，搜索半径 r=4）    │
│          对 F_c 的每个位置 (y,x)，在 F_w 的 (2r+1)²=81 个       │
│          邻域位置计算 dot-product 相似度                         │
│          C = dot(F_c, F_w)  → (1, 81, 128, 128)                │
│                                                                  │
│  Step 4: 迭代更新（训练 4 次 / 推理 12 次）                      │
│          flow_0 = zeros(1,2,128,128)                            │
│                                                                  │
│          for i in range(N):                                      │
│            corr_feat = lookup(C, flow_i)  → (1,81,128,128)     │
│            x = cat([corr_feat, flow_i, context])  (1,115,128,128)│
│            hidden = ConvGRU(hidden, x)    (1,96,128,128)        │
│            Δflow = FlowHead(hidden)       (1,2,128,128)         │
│            flow_{i+1} = flow_i + Δflow                          │
│                                                                  │
│  Step 5: 上采样到输入分辨率                                       │
│          flow_128 (1,2,128,128)                                  │
│          → 双线性插值 × 8 + 偏移量缩放                           │
│          → flow_1024 (1,2,1024,1024)                            │
└──────────────────────────────────────────────────────────────────┘

输出：flow_1024 (1,2,1024,1024)
      flow[:,0,:,:] = dx（水平偏移，像素）
      flow[:,1,:,:] = dy（垂直偏移，像素）
      语义：corrected 坐标 (x,y) → 去 warped 的 (x+dx, y+dy) 采样
```

---

### Stage 2：光流估计（V3 DA-Flow）

**FlowHead V3**（~413万参数，像素 + 扩散特征，参考 DA-Flow）

```
输入：corrected_low  (1,3,1024,1024)
      warped_1024    (1,3,1024,1024)
      q_features     list[4] of (1,16384,3072)  ← DiT Q 特征
      k_features     list[4] of (1,16384,3072)  ← DiT K 特征

┌──────────────────────────────────────────────────────────────────┐
│                     FlowHead V3 DA-Flow                          │
│                                                                  │
│  Step 1: DPT 上采样头（DA-Flow Section 4.4.1）                   │
│          将 DiT 特征从 H/16 上采样到 H/8                         │
│                                                                  │
│          q_features[l]: (1,16384,3072)                          │
│          → reshape: (1,3072,128,128)  [H/8=128]                 │
│          → Conv 1×1: (1,96,128,128)                             │
│          → 4层加权融合（可学习权重）                              │
│          → 双线性上采样 × 2: (1,96,128,128)  ← 已在 H/8         │
│                                                                  │
│          F_Q   = DPT_Q(q_features)    (1,96,128,128)            │
│          F_K   = DPT_K(k_features)    (1,96,128,128)            │
│          F_ctx = DPT_ctx(q_features)  (1,32,128,128)            │
│                                                                  │
│  Step 2: 混合特征编码（DA-Flow Section 4.4.2 Hybrid Encoding）   │
│          CNN 特征提供局部纹理，扩散特征提供几何语义              │
│                                                                  │
│          F_CNN_c = CNN_encoder(corrected)  (1,96,128,128)       │
│          F_CNN_w = CNN_encoder(warped)     (1,96,128,128)       │
│                                                                  │
│          F_hybrid_c = concat(F_CNN_c, F_Q)  (1,192,128,128)    │
│          F_hybrid_w = concat(F_CNN_w, F_K)  (1,192,128,128)    │
│                                                                  │
│  Step 3: Context Encoder（同 V2，只用 corrected）                │
│          hidden (1,96,128,128) + context (1,32,128,128)         │
│                                                                  │
│  Step 4: Correlation Volume（使用混合特征）                       │
│          C = dot(F_hybrid_c, F_hybrid_w)  (1,81,128,128)       │
│                                                                  │
│  Step 5: 迭代更新（与 V2 相同结构）                               │
│          N 次 ConvGRU → flow_1024 (1,2,1024,1024)               │
│                                                                  │
│  核心优势：F_Q / F_K 来自 DiT 的注意力特征，                     │
│  在训练时已学会"看穿"退化和折痕，提供稳健的几何对应信息          │
└──────────────────────────────────────────────────────────────────┘

输出：flow_1024 (1,2,1024,1024)（语义同 V2）
```

---

### Stage 3：高清 Warp

```
输入：flow_1024  (1,2,1024,1024)，推理分辨率下的光流
      orig_img   PIL Image，原始高清 warped 图（2964×2084）

Step 1: upscale_flow（utils/flow_utils.py）
        src_h, src_w = 1024, 1024
        tgt_h, tgt_w = 2084, 2964

        flow_hires[:, 0] = 双线性插值(flow_1024[:, 0]) × (2964/1024)  ← dx 缩放
        flow_hires[:, 1] = 双线性插值(flow_1024[:, 1]) × (2084/1024)  ← dy 缩放
        flow_hires: (1,2,2084,2964)

Step 2: warp_image_with_flow（utils/flow_utils.py）
        将像素偏移转换为归一化坐标：
          dx_norm = flow_hires[:,0] × 2.0 / (2964-1)
          dy_norm = flow_hires[:,1] × 2.0 / (2084-1)

        构建采样 grid：
          grid[y,x] = (identity[y,x][0] + dx_norm[y,x],
                       identity[y,x][1] + dy_norm[y,x])
          grid: (1, 2084, 2964, 2)，值域 [-1, 1]

        F.grid_sample(warped_hires_tensor, grid,
                      mode='bilinear', padding_mode='border',
                      align_corners=True)
        → result: (1,3,2084,2964)

输出：result_hires（PIL Image，2964×2084）← c_*.jpg
```

---

## 训练方案

### 训练数据准备（只做一次）

```
数据格式：metadata_with_flow.csv
  image         corrected_gt 路径（GT 矫正图）
  edit_image    warped 路径（弯曲输入图）
  flow_gt_path  flow_gt.npy 路径（RAFT 计算的反向光流）
  prompt        文本描述

flow_gt.npy 格式：float32，shape (2, 1024, 1024)
  flow_gt[0]：dx（corrected 坐标在 warped 中的水平偏移）
  flow_gt[1]：dy（corrected 坐标在 warped 中的垂直偏移）
  由 generate_flow_labels.py 调用 RAFT 离线计算
```

生成命令：

```bash
CUDA_VISIBLE_DEVICES=6,7,8,9,10,11,12,13,14,15 python utils/generate_flow_labels.py \
    --input_csv  .../1in10_w_metadata.csv \
    --output_csv .../metadata_with_flow.csv \
    --flow_dir   .../flow_labels \
    --gpu_ids 0 1 2 3 4 5 6 7 8 9 \
    --batch_size 4
```

---

### Stage 1：LoRA + FlowHead V2 联合训练

```
┌──────────────────────────────────────────────────────────┐
│  每个训练 step                                            │
│                                                          │
│  数据：corrected_gt, warped, flow_gt, prompt            │
│                                                          │
│  1. Pipeline Units 预处理                                │
│     corrected_gt → VAE encode → input_latents            │
│     warped       → VAE encode → warped_latent（条件）    │
│     prompt       → text encoder → prompt_emb             │
│                                                          │
│  2. L_diffusion（随机 timestep t）                       │
│     z_t = add_noise(input_latents, noise, t)             │
│     noise_pred = DiT(z_t, t, warped_latent, prompt_emb) │
│     L_diff = MSE(noise_pred, target) × weight(t)         │
│     → 更新 LoRA 权重                                     │
│                                                          │
│  3. L_flow + L_warp                                     │
│     corrected_t = preprocess(corrected_gt)  [-1,1]       │
│     warped_t    = preprocess(warped)        [-1,1]       │
│     preds = FlowHeadV2(corrected_t, warped_t, iters=4)  │
│     # preds: list[4] of (1,2,H,W)                       │
│                                                          │
│     flow_gt_scaled = resize(flow_gt, H) × (H/1024)      │
│     L_flow = Σᵢ 0.8^(4-i) × L1(preds[i], flow_gt_s)   │
│                                                          │
│     warp_result = grid_sample(warped_t, preds[-1])       │
│     L_warp = L1(warp_result, corrected_t)                │
│                                                          │
│  4. 总 loss                                             │
│     L = L_diff + 1.0×L_flow + 0.1×L_warp               │
│     → 同时更新 LoRA + FlowHead V2                       │
└──────────────────────────────────────────────────────────┘

启动：bash scripts/train_flow_head.sh
输出：ckpts/step-xxxx.safetensors（含 LoRA keys + flow_head_v2.* keys）
```

---

### Stage 2：FlowHead V3 训练（冻结 DiT）

```
┌──────────────────────────────────────────────────────────┐
│  前提：加载 Stage 1 的 LoRA checkpoint                   │
│        冻结 DiT（含 LoRA），只训练 FlowHead V3           │
│                                                          │
│  每个训练 step                                            │
│                                                          │
│  1. DiT 单步前向（提取 Q/K 特征，不更新梯度）            │
│     随机 timestep t                                      │
│     z_t = add_noise(input_latents, noise, t)             │
│     hook top-4 层 attention 的 Q/K 特征                  │
│     DiT(z_t, t, warped_latent, prompt_emb)  → 触发 hook │
│     q_features: list[4] of (1, T, 3072)                 │
│     k_features: list[4] of (1, T, 3072)                 │
│                                                          │
│  2. FlowHead V3 前向（有梯度）                            │
│     preds = FlowHeadV3(corrected_t, warped_t,           │
│                         q_features, k_features, iters=4) │
│                                                          │
│  3. L_flow + L_warp（同 Stage 1）                        │
│     L = L_diff(冻结，仅计算不更新) + 1.0×L_flow + 0.1×L_warp│
│     → 只更新 FlowHead V3（DPT Head + CNN + GRU）         │
└──────────────────────────────────────────────────────────┘

启动：bash scripts/train_flow_head_v3_daflow.sh
输出：ckpts/step-xxxx.safetensors（含 LoRA keys + flow_head_v3.* keys）
```

---

## 推理方案对比

| 方案 | 脚本 | 光流来源 | 优点 | 缺点 |
|------|------|---------|------|------|
| A：DiT + RAFT | `flow_warp_sample.sh` | RAFT（单独模型）| 光流精度高，基准稳定 | 需加载两个独立模型 |
| B：DiT + FlowHead V2 | `flow_v2_sample.sh` | FlowHead V2 | 单 checkpoint，无 RAFT | 复杂退化区域精度略低于 RAFT |
| C：DiT + FlowHead V3 | `flow_v3_daflow_sample.sh` | FlowHead V3 + DiT 特征 | 扩散特征增强几何感知 | 参数量更大，训练时间更长 |

推理命令（以方案 B 为例）：

```bash
# 修改 scripts/flow_v2_sample.sh：
LORA_PATH="/path/to/step-xxxxx.safetensors"
INPUT_DIR="/path/to/input/image_or_dir"
OUTPUT_DIR="/path/to/output"

bash scripts/flow_v2_sample.sh
```

---

## 版本演进

```
V1（已弃用）
  FlowHead 输入：DiT latent（128×128，16ch）
  问题：latent 无像素匹配机制，误差放大 8 倍，warp 扭曲
  文件：qwen_image_flow_direct.py

    ↓ 改为像素输入

V2（当前主力）
  FlowHead 输入：corrected_low + warped（像素图）
  架构：Feature Encoder → Correlation Volume → ConvGRU × N
  参数：~137 万
  文件：utils/flow_head_v2.py, qwen_image_flow_v2.py

    ↓ 注入 DiT 特征（DA-Flow 思路）

V3（实验版本）
  FlowHead 输入：corrected_low + warped + DiT Q/K 特征
  架构：[DPT Head + CNN] × 2 → Correlation → ConvGRU × N
  参数：~413 万
  文件：utils/flow_head_v3_daflow.py, qwen_image_flow_v3_daflow.py
```

---

## 快速开始

### 训练

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp

# Step 1: 生成光流 GT（只做一次）
CUDA_VISIBLE_DEVICES=6,7,8,9,10,11,12,13,14,15 python utils/generate_flow_labels.py \
    --input_csv  /juicefs-algorithm/data/IPT/yuang_feng/DATA/upwarp_img_1in10_white/1in10_w_metadata.csv \
    --output_csv /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white/metadata_with_flow.csv \
    --flow_dir   /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white/flow_labels \
    --gpu_ids 0 1 2 3 4 5 6 7 8 9 --batch_size 4

# Step 2: 训练 Stage 1（LoRA + FlowHead V2）
bash scripts/train_flow_head.sh

# Step 3（可选）: 修改 LORA_CKPT 后训练 Stage 2（FlowHead V3）
bash scripts/train_flow_head_v3_daflow.sh
```

### 推理

```bash
# 方案 A：DiT + RAFT（修改 LORA_PATH 和 INPUT_DIR）
bash scripts/flow_warp_sample.sh

# 方案 B：DiT + FlowHead V2（修改 LORA_PATH 和 INPUT_DIR）
bash scripts/flow_v2_sample.sh

# 方案 C：DiT + FlowHead V3（修改 LORA_PATH 和 INPUT_DIR）
bash scripts/flow_v3_daflow_sample.sh
```

---

## 文件结构

```
diffsynth_genwarp/
│
├── utils/
│   ├── flow_utils.py               公用：RAFT加载、光流上采样、warp
│   ├── generate_flow_labels.py     离线生成 flow GT（多卡并行）
│   ├── flow_head_v2.py             FlowHead V2 网络（~137万参数）
│   ├── flow_head_v3_daflow.py      FlowHead V3 DA-Flow（~413万参数）
│   └── dpt_head.py                 DPT 上采样头（V3 专用）
│
├── train_flow_head.py              Stage 1 训练（LoRA + V2 联合）
├── train_flow_head_v3_daflow.py    Stage 2 训练（冻结 DiT + V3）
├── train_flow_head_v2.py           V2 独立训练（快速验证）
│
├── qwen_image_flow_warp.py         推理 A：DiT + RAFT
├── qwen_image_flow_v2.py           推理 B：DiT + FlowHead V2
├── qwen_image_flow_v3_daflow.py    推理 C：DiT + FlowHead V3
│
├── scripts/
│   ├── train_flow_head.sh              Stage 1 训练
│   ├── train_flow_head_v3_daflow.sh    Stage 2 训练
│   ├── flow_warp_sample.sh             推理 A 示例
│   ├── flow_v2_sample.sh               推理 B 示例
│   ├── flow_v3_daflow_sample.sh        推理 C 示例
│   └── Acceconfig_8A800.yaml           8×A800 accelerate 配置
│
└── docs/
    └── flow_head_v2_design.md          设计文档（含 DA-Flow 对比分析）
```
