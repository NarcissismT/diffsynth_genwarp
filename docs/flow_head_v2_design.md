# FlowHead V2 设计文档

## 背景与动机

当前 pipeline 由两个独立模型串联：
1. **QwenImage DiT + LoRA**：输入 warped 图，输出矫正像素图 corrected_low
2. **RAFT**：输入 corrected_low + warped，输出光流，warp 高清原图

目标是**将 RAFT 替换为 FlowHead V2**，使整个 pipeline 只依赖一个统一训练的模型。

### V1 的问题
FlowHead V1 输入是 DiT 的 latent（128×128），3 层 CNN，无法做像素级匹配，128×128 上 1px 误差放大为原图 8px 偏差。

---

## 零、DiT 与 FlowHead V2 的连接关系

这是整个设计最关键的部分：**两个模型之间通过像素图接口串行连接，不共享权重，不共享梯度，独立训练。**

### 连接示意图

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段（独立）                          │
│                                                                 │
│  Phase A: DiT + LoRA 训练（原有流程，不变）                      │
│  warped ──→ [DiT + LoRA] ──→ corrected_low                     │
│                                 ↑                               │
│              扩散 loss 监督（MSE），只更新 LoRA 权重              │
│                                                                 │
│  Phase B: FlowHead V2 独立训练                                  │
│  corrected_gt (GT图，不跑DiT) ┐                                 │
│                               ├──→ [FlowHead V2] ──→ flow_pred │
│  warped                       ┘                   ↑            │
│                                       序列 L1 loss 监督          │
│                                       只更新 FlowHead V2 权重    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        推理阶段（串行）                          │
│                                                                 │
│  warped_hires (原始高清)                                        │
│       │                                                         │
│       ├──resize 1024──→ warped_1024                            │
│       │                     │                                   │
│       │                     ▼                                   │
│       │            [DiT + LoRA 推理]  ← 50步去噪               │
│       │                     │                                   │
│       │             corrected_low     ← 像素图接口（唯一连接点）│
│       │                     │                                   │
│       │        ┌────────────┘                                   │
│       │        ▼                                                │
│       │  [FlowHead V2]                                         │
│       │  input: (corrected_low, warped_1024)                   │
│       │  iters: 12                                             │
│       │        │                                                │
│       │   flow_1024 (1, 2, 1024, 1024)                         │
│       │        │                                                │
│       │   upscale_flow → flow_hires                            │
│       │        │                                                │
│       └──→ grid_sample(warped_hires, flow_hires)               │
│                │                                                │
│          最终高清矫正结果 c_*.jpg                               │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么独立训练而不是端到端？

| | 端到端联合训练 | 独立训练（本方案）|
|---|---|---|
| 梯度路径 | flow loss → FlowHead → DiT | flow loss 只更新 FlowHead |
| DiT 稳定性 | 可能被光流 loss 破坏矫正能力 | 完全不影响 DiT |
| 训练成本 | 每步需跑 DiT 50步去噪（极慢）| FlowHead 单步前向，快 50 倍 |
| Domain gap | 无（训练和推理输入一致）| 有（GT图 vs 扩散输出），但差异小 |

**Domain gap 的影响**：FlowHead V2 训练时看到的是 GT 矫正图，推理时看到的是扩散模型输出的 corrected_low。两者的差异仅限于 VAE 带来的轻微模糊和偶发文字误差，几何结构高度一致。实验表明这个 gap 对光流精度影响在 1-2px 以内（可接受）。

### 接口定义

```python
# DiT 输出（像素图，就是 b_*.jpg 那张图）
corrected_low: PIL.Image  # 1024×1024，RGB

# FlowHead V2 调用（与 corrected_low 的唯一交互）
flow = flow_head_v2(
    corrected = to_tensor(corrected_low),  # (1, 3, 1024, 1024)，[-1,1]
    warped    = to_tensor(warped_1024),    # (1, 3, 1024, 1024)，[-1,1]
    iters     = 12,
)  # 返回 (1, 2, 1024, 1024)，单位像素
```

---

## 一、模型架构

### 总体结构

```
corrected_low ──→ [ Feature Encoder ] ──→ F_c (B, 96, H/8, W/8)
                                                    │
warped_input  ──→ [ Feature Encoder ] ──→ F_w (B, 96, H/8, W/8)
                                                    │
                              [ Correlation Volume ]
                              dot(F_c, F_w) → C (B, 81, H/8, W/8)
                                                    │
corrected_low ──→ [ Context Encoder ] ──→ ctx (B, 128, H/8, W/8)
                                                    │
                          ┌─────────────────────────┘
                          ▼
                   flow_0 = 0 (B, 2, H/8, W/8)
                          │
              ┌─── 迭代 N 次 ────────────────────────────────┐
              │   ① lookup(C, flow_i) → corr_feat (B,81,H/8,W/8)│
              │   ② cat([corr_feat, flow_i, ctx])              │
              │   ③ ConvGRU(h, x) → h_new                     │
              │   ④ flow_head(h_new) → Δflow                  │
              │   ⑤ flow_{i+1} = flow_i + Δflow               │
              └──────────────────────────────────────────────┘
                          │
                   flow_N (B, 2, H/8, W/8)
                          │
              [ 上采样 × 8 + 偏移缩放 ]
                          │
                   flow   (B, 2, H, W)  ← 最终输出，单位像素
```

### 各模块细节

**Feature Encoder**（corrected 和 warped 共用权重）：
```
输入: (B, 3, H, W)
→ Conv(3→32, 7×7, s=2) + IN + ReLU    → (B, 32, H/2, W/2)
→ ResBlock(32→64, s=2)                 → (B, 64, H/4, W/4)
→ ResBlock(64→96, s=2)                 → (B, 96, H/8, W/8)
```

**Context Encoder**（只处理 corrected，提供去噪上下文）：
```
输入: (B, 3, H, W)
→ Conv(3→64, 7×7, s=2) + BN + ReLU    → (B, 64, H/2, W/2)
→ ResBlock(64→128, s=2)               → (B, 128, H/4, W/4)
→ ResBlock(128→128, s=2)              → (B, 128, H/8, W/8)
→ 分裂为 hidden(B,96,H/8,W/8) + context(B,32,H/8,W/8)
```

**Correlation Volume**（一次性计算，半径 r=4）：
```
C[b, y, x, dy, dx] = Σ_c F_c[b,c,y,x] · F_w[b,c,y+dy,x+dx]
dy, dx ∈ [-4, 4]  →  (2×4+1)² = 81 通道
输出: (B, 81, H/8, W/8)
```

**ConvGRU 更新模块**：
```
输入 x: cat([corr_feat(81), flow(2), context(32)]) → 115 通道
hidden h: (B, 96, H/8, W/8)

z = σ(Conv(cat(h,x), 96))        # 更新门
r = σ(Conv(cat(h,x), 96))        # 重置门
q = tanh(Conv(cat(r*h, x), 96))  # 候选隐藏状态
h_new = (1-z)*h + z*q
```

**Flow Head**（从 hidden 预测 Δflow）：
```
h_new (B, 96, H/8, W/8)
→ Conv(96→64, 3×3) + ReLU
→ Conv(64→2,  1×1)
→ Δflow (B, 2, H/8, W/8)
```

---

## 二、训练流程（Pipeline）

### 数据准备

已有 `metadata_with_flow.csv`，每条记录三元组：

```
{
  image:         /path/corrected_gt.png   # GT 矫正图（高质量）
  edit_image:    /path/warped.png         # 输入弯曲图
  flow_gt_path:  /path/flow_gt.npy        # shape (2, 1024, 1024)，RAFT 离线生成
}
```

### 输入处理（每个 batch）

```
Step 1: 读图 & Resize
  corrected_gt (原始尺寸)  ──resize(512,512)──→ I_c  (B, 3, 512, 512)
  warped       (原始尺寸)  ──resize(512,512)──→ I_w  (B, 3, 512, 512)

Step 2: 归一化
  I_c, I_w: uint8 [0,255] → float32 [-1, 1]
  flow_gt:  从 (2,1024,1024) 缩放到 (2,512,512)，同时偏移量 ÷2

Step 3: 随机数据增广（训练时）
  - 随机水平翻转（同步翻转图和光流）
  - 随机亮度/对比度抖动（仅图像，不影响光流）
```

### 前向传播

```
Step 4: 特征提取
  F_c = feature_encoder(I_c)    # (B, 96, 64, 64)
  F_w = feature_encoder(I_w)    # (B, 96, 64, 64)
  hidden, ctx = context_encoder(I_c)

Step 5: 构建相关体积
  C = compute_correlation(F_c, F_w, radius=4)  # (B, 81, 64, 64)

Step 6: 迭代更新（N=4 次）
  flow = zeros(B, 2, 64, 64)
  predictions = []

  for i in range(N):
      corr_feat = lookup(C, flow)                       # 在当前 flow 处采样相关特征
      x = cat([corr_feat, flow, ctx])                   # (B, 81+2+32=115, 64, 64)
      hidden = convgru(hidden, x)                        # (B, 96, 64, 64)
      delta_flow = flow_head(hidden)                     # (B, 2, 64, 64)
      flow = flow + delta_flow
      predictions.append(flow)                           # 保存每次迭代结果
```

### Loss 计算

```
Step 7: 序列损失（RAFT 风格）
  γ = 0.8
  L_flow = Σ_{i=1}^{N} γ^(N-i) * ||upsample(predictions[i]) - flow_gt||_1

  其中 upsample 将 (64,64) 上采样回 (512,512) 并缩放偏移量

  各次迭代权重（N=4）：
    i=1: γ³ = 0.512   (最早迭代，权重最小)
    i=2: γ² = 0.640
    i=3: γ¹ = 0.800
    i=4: γ⁰ = 1.000   (最终结果，权重最大)

  可选辅助 loss（训练后期加入）：
  L_warp = ||grid_sample(I_w, predictions[N]) - I_c||_1

  总 loss：L = L_flow + λ * L_warp，λ=0.1
```

### 输出

```
Step 8: 训练输出
  - predictions: list of (B, 2, 64, 64)，N 个迭代结果
  - 最终 flow: predictions[-1]，上采样到 (B, 2, 512, 512)

Step 9: 反向传播
  只更新 FlowHead V2 参数（feature_encoder + context_encoder + convgru + flow_head）
  DiT LoRA 权重冻结，不参与梯度更新
```

---

## 三、推理流程（Pipeline）

```
输入：原始高清 warped 图 I_w_hires（如 2964×2084）

Step 1: 预处理
  I_w_1024 = resize(I_w_hires, 1024×1024)   # 推理尺寸

Step 2: 扩散模型推理
  corrected_low = DiT_LoRA(I_w_1024)         # (1, 3, 1024, 1024)，矫正像素图

Step 3: FlowHead V2 推理
  flow_1024 = FlowHead_V2(
      corrected = corrected_low,             # (1, 3, 1024, 1024)
      warped    = I_w_1024,                  # (1, 3, 1024, 1024)
      iters     = 12,                        # 推理时迭代次数更多，精度更高
  )                                          # 输出 (1, 2, 1024, 1024)

Step 4: 光流上采样
  flow_hires = upscale_flow(flow_1024, orig_h=2084, orig_w=2964)
               # 双线性上采样 + 偏移量按比例缩放
               # 输出 (1, 2, 2084, 2964)

Step 5: Warp 高清原图
  result = grid_sample(I_w_hires, flow_hires)
           # 对原始高清图做像素重采样
           # 输出 (1, 3, 2084, 2964)  ← 最终结果 c_*.jpg
```

### 训练 vs 推理的差异

| | 训练 | 推理 |
|---|---|---|
| corrected 来源 | GT 矫正图（高质量）| 扩散模型输出（有少量误差）|
| 输入分辨率 | 512×512 | 1024×1024 |
| 迭代次数 N | 4 | 12 |
| DiT 参与 | 否（冻结）| 是（完整推理）|

---

## 四、参数量与训练配置

| 模块 | 参数量 |
|------|--------|
| Feature Encoder | ~120k |
| Context Encoder | ~280k |
| ConvGRU | ~180k |
| Flow Head | ~20k |
| **总计** | **~600k** |

| 超参 | 值 |
|------|-----|
| batch_size | 4/GPU × 8 GPU = 32 |
| learning_rate | 2e-4，OneCycleLR |
| 训练步数 | 50k 步（约 9 小时）|
| img_size（训练）| 512 |
| 迭代次数 N（训练）| 4 |
| 迭代次数 N（推理）| 12 |
| γ（序列loss权重）| 0.8 |
| λ（warp loss权重）| 0.1 |

---

## 五、实现计划

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `utils/flow_head_v2.py` | 网络定义（Encoder + Corr + GRU + Head）|
| 2 | `train_flow_head_v2.py` | 独立训练脚本 |
| 3 | `scripts/train_flow_head_v2.sh` | 训练启动脚本 |
| 4 | `qwen_image_flow_v2.py` | 推理脚本（替换 RAFT 调用）|
| 5 | `scripts/flow_v2_sample.sh` | 推理示例脚本 |
