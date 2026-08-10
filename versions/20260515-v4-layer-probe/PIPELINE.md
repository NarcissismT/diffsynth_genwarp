# V4 Layer Probing — 训练与推理流程详解

文档生成于 2026-05-26，对应 ckpt：
`/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/`

---

## 目录

- [一、整体设计意图](#一整体设计意图)
- [二、训练流程](#二训练流程)
  - [2.1 数据](#21-数据)
  - [2.2 模型组件](#22-模型组件)
  - [2.3 训练 forward](#23-训练-forward)
  - [2.4 Loss](#24-loss)
  - [2.5 ckpt 保存](#25-ckpt-保存)
  - [2.6 训练参数总览](#26-训练参数总览)
- [三、推理流程](#三推理流程)
  - [3.1 推理脚本结构](#31-推理脚本结构)
  - [3.2 推理特征提取（必须与训练一致）](#32-推理特征提取必须与训练一致)
  - [3.3 推理 FlowHead 调用](#33-推理-flowhead-调用)
  - [3.4 推理输出](#34-推理输出)
- [四、训练-推理一致性对照表](#四训练-推理一致性对照表)
- [五、已知问题与历史 Bug](#五已知问题与历史-bug)
- [六、文件索引](#六文件索引)

---

## 一、整体设计意图

V4 是 **DA-Flow 风格**的文档矫正光流网络。核心思想：

```
弯曲文档 (warped)
  ├─→ Qwen 扩散完整推理 → 矫正图 (corrected_low)，但是文字会损失（VAE 瓶颈）
  └─→ Qwen DiT 中间层 Q/K → 几何先验特征
       │
       ↓
     FlowHead V4：(corrected, warped, q_feats, k_feats) → backward flow
       │
       ↓
     用 flow 对原图做像素重采样 → 文字保留原始像素的高清矫正图
```

V4 相对 V3 的唯一升级：**`--dit_target_layers` 任意指定 DiT 层组合**（不限于"最后 N 层"），用于 layer probing 实验找最优层。

---

## 二、训练流程

### 2.1 数据

**数据集路径**：`/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/`

每条样本由 CSV 行索引：

| 字段 | 内容 | 文件示例 |
|------|------|----------|
| `image` | rectified GT（GT 矩形矫正图）| `gt/Doc3d_crop/Doc3d_crop_0000000.png` |
| `edit_image` | warped（弯曲输入图）| `img/Doc3d_crop/Doc3d_crop_0000000.png` |
| `prompt` | 矫正 prompt（文本指令）| "Apply geometric correction..." |
| `category` | 数据集名 | Doc3d_crop / Xm_crop / Pers_NoAug / ... |
| `flow_gt_path` | 反向 flow numpy 文件 | `flow/Doc3d_crop/Doc3d_crop_0000000.npy` |

**分辨率**：所有图像 **512×512**；flow_gt 是 **(2, 1024, 1024)**（在 train forward 时被插值到模型实际预测分辨率）。

**flow_gt 语义**：backward flow，`flow[y,x] = (dx,dy)` 表示矫正图 `(x,y)` 应到弯曲图 `(x+dx, y+dy)` 处采样。

**训练数据规模**：约 116k 条样本，6 类 category。

---

### 2.2 模型组件

#### 2.2.1 Qwen Pipeline（完全冻结）

```
QwenImagePipeline:
  text_encoder    （Qwen2.5-VL，3584 维）
  dit             （60 层 dual-stream DiT，3072 维）
    + LoRA      （rank=32，加载 step-668000.safetensors）
  vae             （图像编码/解码，4× 下采样）
  scheduler       （FlowMatchScheduler，1000 timesteps）
```

**冻结策略**：训练时 `freeze_dit=True`（默认），所有 Qwen 参数 `requires_grad=False`。LoRA 权重也加载但不更新。

#### 2.2.2 FlowHead V4（唯一可训练模块）

定义在 [utils/flow_head_v4_layer_probe.py](utils/flow_head_v4_layer_probe.py)：

```
FlowHeadV4LayerProbe(
  iters=4,           # GRU 迭代次数
  radius=4,          # CorrBlock 搜索半径
  dit_channels=3072, # DiT 隐层维度
  diff_out_ch=96,    # DPT 输出通道
  cnn_out_ch=96,     # CNN 输出通道
  num_dit_layers=L,  # 等于 len(dit_target_layers)
)
├── dpt_heads        # DPTHeads(num_layers=L)
│   ├── dpt_q        # DPT 头：Q 特征上采样
│   ├── dpt_k        # DPT 头：K 特征上采样
│   └── dpt_ctx      # DPT 头：context 特征（暂未使用）
├── feat_encoder     # HybridEncoder：CNN + DPT 拼接
├── context_encoder  # ContextEncoder：corrected 图初始化 GRU hidden（含 BatchNorm）
├── update_block     # ConvGRU：迭代更新 hidden
└── flow_head        # FlowPredHead：hidden → Δflow 输出头
```

**参数量**：单层 ~206 万，三层 ~344 万。

#### 2.2.3 dit_target_layers 选项

| 实验 | 层（0-based）| 描述 |
|------|------------|------|
| A | `"11"` | Layer 12，早层 |
| B | `"23"` | Layer 24，中浅层 |
| C | `"35"` | Layer 36，中层（推荐主候选）|
| D | `"47"` | Layer 48，中后层 |
| E | `"23,35,47"` | 三层融合 |
| F | `"11,23,35,47"` | 四层融合 |

---

### 2.3 训练 forward

代码：[train_flow_head_v4_layer_probe.py](train_flow_head_v4_layer_probe.py)

```
def forward(data):
  ┌───────────────────────────────────────┐
  │ Step A: forward_preprocess            │
  │   pipe.units 流水线执行：              │
  │   - input_image = warped (edit_image)  │
  │   - VAE encode warped → input_latents  │
  │   - 生成 noise = randn_like(latents)   │
  │   - prompt → prompt_emb (经过           │
  │       extract_masked_hidden + drop +    │
  │       pad 处理)                         │
  └───────────────────────────────────────┘
              ↓
  ┌───────────────────────────────────────┐
  │ Step B: _extract_qk_features (DiT 单步) │
  │   1. 随机采样 timestep ∈ [400, 600)     │
  │   2. add_noise(input_latents, noise, t) │
  │      → noisy_latent                    │
  │   3. DiT 单步前向：                      │
  │      img_in → 60 层 transformer_blocks │
  │   4. 在 dit_target_layers 指定层提取 Q/K │
  │      （从 attn 之前的 normed+modulated  │
  │       img tokens 取 to_q/to_k 投影）    │
  │   5. 在 max(target_layers) 后 break    │
  │   全程 torch.no_grad()                  │
  └───────────────────────────────────────┘
              ↓
  ┌───────────────────────────────────────┐
  │ Step C: PIL → tensor                   │
  │   corrected_t = preprocess(GT)         │
  │   warped_t    = preprocess(edit_image) │
  │   值域 [-1, 1]                          │
  └───────────────────────────────────────┘
              ↓
  ┌───────────────────────────────────────┐
  │ Step D: FlowHead V4 前向 (iters=4)     │
  │   1. dpt_heads(q,k,(H,W))              │
  │      → F_Q, F_K (B, 96, H/8, W/8)      │
  │   2. fmap_c = feat_encoder(corrected,F_Q)│
  │      fmap_w = feat_encoder(warped,   F_K)│
  │   3. context_encoder(corrected)         │
  │      → hidden (96 ch), context (32 ch) │
  │   4. CorrBlock(fmap_c, fmap_w, r=4)    │
  │   5. for _ in range(4):                │
  │        flow.detach()                    │
  │        corr_feat = corr_block(flow)    │
  │        hidden = update_block(hidden,    │
  │          [corr_feat, flow, context])    │
  │        Δflow  = flow_head(hidden)       │
  │        flow   = flow + Δflow            │
  │        upsample to (H,W) into preds[]  │
  │   返回 list of 4 个预测                 │
  └───────────────────────────────────────┘
              ↓
  ┌───────────────────────────────────────┐
  │ Step E: 计算 Loss                       │
  │   见 2.4                                │
  └───────────────────────────────────────┘
```

---

### 2.4 Loss

```python
# flow_gt 缩放到预测分辨率 (H/8 × H/8)
fh = predictions[-1].shape[-2]
scale = fh / flow_gt.shape[-2]   # 1/8
flow_gt_resized = interpolate(flow_gt, (fh,fh)) * scale

# L1：序列 loss（4 次迭代加权 L1，gamma=0.8）
L_flow = sequence_loss(predictions, flow_gt_resized, gamma=0.8)
       = sum_{i=0..3} gamma^(N-1-i) * L1(pred_i, gt)

# L_warp：用最终预测 flow 把 warped warp 到 corrected
flow_final = predictions[-1]
warp_result = grid_sample(warped_t_resized, flow_final)
L_warp = L1(warp_result, corrected_t_resized)

# 总 loss
loss = lambda_flow * L_flow + lambda_warp * L_warp
     = 1.0 * L_flow + 0.5 * L_warp
```

**注意**：
- `freeze_dit=True` 默认开启，**没有 L_diffusion**（不参与扩散去噪 loss）
- 没有 smoothness loss（V4 没加）
- 没有 fold/Jacobian regularization

---

### 2.5 ckpt 保存

**只保存可训练参数**（即 FlowHead V4 参数，DiT 冻结不保存）：

```python
ModelLogger(output_path, remove_prefix_in_ckpt="flow_head.")
```

`remove_prefix_in_ckpt="flow_head."` 会把训练模型的 `pipe.dit` 之外、模块名以 `flow_head.` 开头的 key 去掉这个前缀。**实际效果**：

| 模型 attribute 路径 | ckpt 里的 key |
|--------------------|--------------|
| `self.flow_head.context_encoder.layer1.conv1.weight` | `context_encoder.layer1.conv1.weight` ✓ |
| `self.flow_head.feat_encoder.layer2.conv1.weight` | `feat_encoder.layer2.conv1.weight` ✓ |
| `self.flow_head.flow_head.conv1.weight` (FlowPredHead) | `flow_head.conv1.weight` ✓ |
| `self.flow_head.dpt_heads.dpt_q.projs.0.weight` | `dpt_heads.dpt_q.projs.0.weight` ✓ |

**注意**：
- BatchNorm 的 `running_mean / running_var / num_batches_tracked` **不在 ckpt 里**（因为 `requires_grad=False`），`load_state_dict(strict=False)` 加载时缺失。这是一个**已知遗留问题**——会用初始 BN 统计量（mean=0, var=1），可能影响 ContextEncoder 的稳定性。

**ckpt 大小**：
- 单层（实验 C）：~7.9 MB
- 三层（实验 E）：~14 MB

---

### 2.6 训练参数总览

来自 [scripts/train_exp_C_layer36.sh](scripts/train_exp_C_layer36.sh)：

```bash
--max_pixels 1048576              # 1024^2，但实际数据是 512×512 不被 cap
--dataset_repeat 5                # 每个 epoch 重复 5 次
--num_epochs 6
--learning_rate 1e-4
--save_steps 4000
--remove_prefix_in_ckpt "flow_head."

--lora_checkpoint <step-668000.safetensors>   # 加载 DiT LoRA（冻结后用于特征提取）
--dit_target_layers "35"          # 该实验取的层
--diff_out_ch 96
--lambda_flow 1.0
--lambda_warp 0.5
```

**总训练步数估算**：116k samples × 5 repeat × 6 epoch ≈ 348 万步（batch=1）。

---

## 三、推理流程

### 3.1 推理脚本结构

主脚本：[qwen_image_flow_v4.py](qwen_image_flow_v4.py)
启动脚本：[scripts/flow_v4_sample.sh](scripts/flow_v4_sample.sh)

```
process_images(args)
  ├── load_pipeline(args, device)
  │     1. QwenImagePipeline.from_pretrained
  │     2. add_lora_to_model + 加载 step-668000 LoRA
  │     3. _patch_pipeline_for_v4(pipe)：monkey-patch __call__
  │     4. enable_vram_management
  │     5. FlowHeadV4LayerProbe + 加载 ckpt
  │
  └── for each img_path:
        1. 读取 + resize 到 IMG_SIZE (默认 512×512)
        2. pipe(prompt, edit_image, ..., num_inference_steps=50)
           - 走原版 __call__ 完整扩散推理 → corrected_low
           - 走 patched 部分提取 Q/K（见 3.2）
           - 返回 (corrected_low, q_features, k_features)
        3. flow_model(corrected_t, warped_t, q_feats, k_feats, iters=12)
        4. upscale_flow → 原图分辨率
        5. warp_image_with_flow → 高清矫正图
        6. 保存 a/b/c/d 四张图
```

---

### 3.2 推理特征提取（必须与训练一致）

`_patch_pipeline_for_v4` 在 `pipe.__call__` 上挂的 `patched_call` 流程：

```
patched_call(prompt, edit_image, num_inference_steps, height, width, ...):
  ┌────────────────────────────────────────────┐
  │ Stage 1: Qwen 完整扩散推理                   │
  │   self.extract_daflow_features = False      │
  │   image = original_call(...)               │ ← Qwen 50 步去噪
  │   self.extract_daflow_features = True       │
  │   image = corrected_low (PIL)               │
  └────────────────────────────────────────────┘
              ↓
  ┌────────────────────────────────────────────┐
  │ Stage 2: 重新走 pipe.units 准备 inputs       │
  │   （与训练 forward_preprocess 完全一致！）   │
  │   inputs_shared.input_image = edit_image     │
  │     → input_latents 来自 warped              │
  │   pipe.units 流水线产出：                    │
  │     - input_latents (warped 的 VAE latent)   │
  │     - noise        (NoiseInitializer 生成)   │
  │     - prompt_emb   (extract_masked_hidden    │
  │                     + drop + pad 处理过的)   │
  │     - prompt_emb_mask                        │
  └────────────────────────────────────────────┘
              ↓
  ┌────────────────────────────────────────────┐
  │ Stage 3: 加噪到固定 t = 500（与训练对齐）     │
  │   timestep = scheduler.timesteps[500]       │
  │   noisy_latent = add_noise(input_latents,   │
  │                            noise, t=500)    │
  └────────────────────────────────────────────┘
              ↓
  ┌────────────────────────────────────────────┐
  │ Stage 4: DiT 单步前向，提取目标层 Q/K        │
  │   image_tokens = rearrange + img_in         │
  │   text_tokens  = txt_in(txt_norm(prompt_emb))│
  │   conditioning = time_text_embed(t=500)     │
  │   for block_idx, block in transformer_blocks:│
  │     if block_idx in target_layers:          │
  │       提取 Q/K（与训练完全相同）             │
  │     执行 block forward                      │
  │     if block_idx == max_target_layer: break │
  │   全程 torch.no_grad()                       │
  └────────────────────────────────────────────┘
              ↓
  return (corrected_low, q_features, k_features)
```

---

### 3.3 推理 FlowHead 调用

```python
# 加载 ckpt（修复后的版本，不再多余 replace）
sd = load_file(ckpt_path)
flow_model.load_state_dict(sd, strict=False)
# 缺失：仅 BN buffer（21 个 running_mean/var/num_batches_tracked）
# 多余：0
# 关键参数全部加载

# RGB 通道使用 corrected_low（不是 GT，因为推理没有 GT）
corrected_t = pil_to_tensor(corrected_low)   # Qwen 扩散输出
warped_t    = pil_to_tensor(img_input)        # 推理输入图

with torch.no_grad():
    flow_low = flow_model(
        corrected_t, warped_t,
        q_features, k_features,
        iters=12,    # 推理用 12 次（训练只有 4 次）
    )
# flow_low shape: (1, 2, IMG_SIZE, IMG_SIZE)，单位像素
```

**注意**：训练 RGB 通道用 GT rectified，推理用 corrected_low。这是必要的折中——因为推理时没有 GT。corrected_low 的内容大致接近 GT 但有 VAE 瓶颈带来的细节损失。

---

### 3.4 推理输出

每张输入图生成 4 张：

| 文件 | 含义 |
|------|------|
| `xxx_a.jpg` | 推理输入（resize 后的 warped）|
| `xxx_b.jpg` | Qwen 扩散输出 corrected_low（仅供参考）|
| `xxx_c.jpg` | **最终结果**：用 flow warp 原图 → 高清矫正图（文字来自原图像素）|
| `xxx_d.jpg` | 三合一对比：原始 \| 扩散 \| V4 warp |

**几何放大**：
```
flow_low (1, 2, IMG_SIZE, IMG_SIZE)
  → upscale_flow(target_h, target_w)  ← 双线性插值 + 数值按比例缩放
  → flow_hires (1, 2, orig_h, orig_w)
  → grid_sample(原始 RGB, identity_grid + flow_hires)
  → result_hires
```

---

## 四、训练-推理一致性对照表

下表是当前修复完成后的对齐状态：

| 项目 | 训练 | 推理（修复后）| 一致 |
|------|------|------|------|
| input_image (latent 来源) | warped (edit_image) | warped (edit_image) | ✓ |
| input_latents 提取方式 | pipe.units 流水线 | pipe.units 流水线 | ✓ |
| prompt_emb 处理 | extract_masked_hidden + drop + pad | extract_masked_hidden + drop + pad | ✓ |
| timestep | 随机 [400, 600) | 固定 500 | ✓（数值范围对齐）|
| add_noise 输入 | input_latents + noise + t | input_latents + noise + t | ✓ |
| DiT timestep embedding | timestep ≈ 500 | timestep == 500 | ✓ |
| Q/K 提取层 | dit_target_layers | _v4_target_layers (相同集合) | ✓ |
| 提前 break | max(target_layers) | max(target_layers) | ✓ |
| FlowHead RGB 输入 (corrected) | GT rectified | corrected_low（Qwen 扩散输出）| ✗ 必要折中 |
| FlowHead RGB 输入 (warped) | warped | warped | ✓ |
| FlowHead iters | 4 | 12 | ✗ 推理多 8 次 refine |
| 输入图分辨率 | 512×512 | IMG_SIZE=512×512 | ✓ |
| FlowHead 预测分辨率 | (H/8, W/8)=64×64 | 同 | ✓ |

---

## 五、已知问题与历史 Bug

按发现顺序排列。

### Bug #1（已修）：lora_checkpoint 加载失败

[train_flow_head_v4_layer_probe.py:144-159](train_flow_head_v4_layer_probe.py#L144-L159) 之前直接 `pipe.dit.load_state_dict(sd, strict=False)` 没先加 LoRA，导致 LoRA key 全部被静默跳过。

**修复**：先 `add_lora_to_model + mapping_lora_state_dict`，再 `load_state_dict`。

### Bug #2（已修）：forward_preprocess 用错图

之前 `input_image=data["image"]`（GT rectified），导致 `input_latents` 来自 GT 而不是 warped。

**修复**：改为 `input_image = data.get("edit_image") or data["image"]`，优先用 warped。

### Bug #3（已修）：推理 sys.path 缺失主项目根目录

versions/ 子目录下的脚本运行时找不到 `utils.flow_utils`。

**修复**：在 sys.path 加入 `_PROJECT_ROOT`。

### Bug #4（已修）：推理 evaluate_real 输入颠倒

之前推理时 `flow_model(warped, warped, ...)`，corrected 自己和自己比 → flow ≈ 0 → 图像几乎不变。

**修复**：改为 `flow_model(corrected_low, warped, ...)`，用 Qwen 扩散输出作为参考。

### Bug #5（已修）：推理 prompt_emb 处理不一致

之前推理 `patched_call` 自己手动调用 `text_encoder` 取 `hidden[-1]`，没经过 `extract_masked_hidden + drop + pad`，与训练 `pipe.units` 输出的 `prompt_emb` 内容/长度/对齐都不同。

**修复**：推理 `patched_call` 直接复用 `pipe.units` 流水线。

### Bug #6（已修）：推理 ckpt 多余 replace 导致 FlowPredHead 用随机权重

之前推理 `sd.replace("flow_head.", "", 1)` 把内层 FlowPredHead 的 `flow_head.conv1` 也剥成 `conv1`，导致：
- 模型期待 `flow_head.conv1.weight`（FlowPredHead 输出层）
- ckpt 里却是 `conv1.weight` → 加载时缺失 → **FlowPredHead 用随机权重**

这是导致**水波纹崩坏**的真正主因——推理时 GRU 的 hidden 经过随机 conv 输出随机 Δflow，12 次迭代累积成大幅错乱位移。

**修复**：删除推理时多余的 `replace` 调用，直接 `load_state_dict`。

### 已知遗留问题

- **BN buffer 未保存**：21 个 BatchNorm 的 running_mean/var/num_batches_tracked 在 ckpt 里缺失（`export_trainable_state_dict` 只保 `requires_grad=True` 的参数）。推理时用初始统计量，不理想但能跑。彻底修复需要 ModelLogger 同时保存 buffer 或训练时切到 `track_running_stats=False`。

- **训练样本无负样本/低畸变样本**：训练数据 |flow|.mean ≈ 10-30 px，模型未见过 |flow| ≈ 0 的样本。如果输入图本身就基本是矩形（例如 silver_bullet 测试集很多图），模型仍会预测一定幅度 flow。

- **缺 smoothness 和 fold 正则**：method.md 提到的 L_smooth + L_fold 都没加。

- **推理 iters=12 vs 训练 iters=4**：推理多 8 次 refine 是 V3 沿用的设定，未必最优。

---

## 六、文件索引

```
versions/20260515-v4-layer-probe/
├── README.md                              # 总览（实验设计）
├── PIPELINE.md                            # 本文档
├── train_flow_head_v4_layer_probe.py      # 训练主脚本
├── qwen_image_flow_v4.py                  # 推理主脚本
├── eval_flow_v4_quant.py                  # 定量评测脚本
├── utils/
│   └── flow_head_v4_layer_probe.py        # FlowHead V4 模型定义
└── scripts/
    ├── train_exp_A_layer12.sh ~ F         # 6 个训练实验
    ├── flow_v4_sample.sh                  # 推理启动脚本
    └── eval_exp_compare.sh                # 6 实验对比评测
```

主项目副本（实际跑训练时用的）：
```
/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/
├── train_flow_head_v4_layer_probe.py
├── qwen_image_flow_v4.py
├── utils/flow_head_v4_layer_probe.py
└── scripts/train_exp_*.sh
```

ckpt 路径：
```
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/
├── 20260515-exp_C_layer36/                # 单层 Layer 36
│   └── step-{4000~240000}.safetensors
└── 20260515-exp_E_layer24_36_48/          # 三层融合
    └── step-{4000~228000}.safetensors
```

DiT LoRA：
```
/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/
└── 20250929-1_1in10_w_unwarp/
    └── step-668000.safetensors
```

数据集：
```
/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/
├── metadata_with_flow.csv                 # 训练 manifest（116k 行）
├── gt/<category>/*.png                    # rectified GT (512×512)
├── img/<category>/*.png                   # warped (512×512)
└── flow/<category>/*.npy                  # flow_gt (2, 1024, 1024)
```

真实测试集：
```
/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp/
└── *.jpg  (2564 张真实文档图，无 GT flow)
```
