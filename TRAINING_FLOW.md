# diffsynth_genwarp 训练流程说明

> 本文档总结当前仓库实际在跑的训练管线，便于快速上手与交接。
> 所有相对路径均基于仓库根目录 `diffsynth_genwarp/`。

---

## 0. 端到端 Pipeline Overview（从输入到输出）

本节描述一次训练 step 的完整数据流。**核心任务形态**：给定一张"受扰动"的图像 `edit_image`（例如 warp 后的图）与文本 `prompt`，学习预测"目标图像" `image` 的 flow-matching 速度场；推理时即可用训练好的 LoRA 把 warped 图像 unwarp 回目标图像。

### 0.1 一次训练 step 的数据流（ASCII Pipeline）

```
┌──────────────────────────────────────────────────────────────────┐
│  CSV metadata (image, edit_image, prompt)                        │
│        │                                                          │
│        ▼                                                          │
│  ImageDataset.__getitem__                                         │
│    • 读 PIL 图像 + prompt                                         │
│    • crop_and_resize 到合法分辨率（受 max_pixels 约束）           │
│    • 返回 {"prompt", "image", "edit_image"}                       │
└──────────────────────────────────────────────────────────────────┘
                               │  data (dict of PIL/str)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  QwenImageTrainingModule.forward_preprocess(data)                 │
│    依次跑 self.pipe.units（详见 0.2），得到：                     │
│      input_latents     ← VAE.encode(image)          目标          │
│      edit_latents      ← VAE.encode(edit_image)     条件          │
│      noise             ← N(0, I), shape=input_latents             │
│      prompt_emb,       ← Qwen2.5-VL text encoder                  │
│      prompt_emb_mask      (Qwen-Image-Edit 模板里含 <image_pad>)  │
│      height, width                                                │
└──────────────────────────────────────────────────────────────────┘
                               │  inputs (dict of tensors)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  pipe.training_loss(**models, **inputs)                           │
│    1) 随机采 timestep t ∈ [0, 1000)                               │
│    2) x_t   = (1-σ) * input_latents + σ * noise   （加噪）        │
│    3) target = noise - input_latents              （FM 速度场）   │
│    4) noise_pred = model_fn_qwen_image(dit, latents=x_t, ...)     │
│    5) loss = MSE(noise_pred, target) * w(t)                       │
└──────────────────────────────────────────────────────────────────┘
                               │  loss (scalar)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  accelerator.backward(loss) → optimizer.step()                    │
│  仅更新 LoRA 参数；每 save_steps 保存一次 .safetensors            │
└──────────────────────────────────────────────────────────────────┘
```

### 0.2 `pipe.units` 逐步展开（实际执行的 8 个单元）

定义于 [diffsynth/pipelines/qwen_image.py:66-75](diffsynth/pipelines/qwen_image.py#L66-L75)，按顺序由 `PipelineUnitRunner` 跑完。每个单元声明 `input_params` 和 `onload_model_names`，Runner 负责按需把对应子模型 load 到 GPU。

| # | Unit | 干了什么 | 产出 |
| --- | --- | --- | --- |
| 1 | `ShapeChecker` | 把 `height/width` 对齐到 `16` 的倍数 | `height, width` |
| 2 | `NoiseInitializer` | `torch.randn((1, 16, H/8, W/8))`，作为扩散起点 | `noise` |
| 3 | `InputImageEmbedder` | VAE 编码 `image` → latent（训练时仅返回，不提前加噪） | `input_latents`（训练目标） |
| 4 | `Inpaint` | 无 mask 时直接跳过 | — |
| 5 | `PromptEmbedder` | **关键**：当 `edit_image` 存在时切换到 Qwen-Image-Edit 模板，使用 `Qwen2VLProcessor` 把图像一起塞入 text encoder（`<vision_start><image_pad><vision_end>{prompt}`），产出多模态 `prompt_emb` | `prompt_emb, prompt_emb_mask` |
| 6 | `EntityControl` | 训练脚本未传 entity 参数，跳过 | — |
| 7 | `BlockwiseControlNet` | 训练脚本未启用，跳过 | — |
| 8 | `EditImageEmbedder` | VAE 编码 `edit_image` → `edit_latents`（条件 latent） | `edit_latents` |

### 0.3 `model_fn_qwen_image`：DiT 前向的关键动作

实现于 [diffsynth/pipelines/qwen_image.py:599-675](diffsynth/pipelines/qwen_image.py#L599-L675)。它决定了条件（`edit_latents`）是如何与噪声 latent 交互的：

1. **Patchify**：把噪声 latent `[B, 16, H/8, W/8]` 做 2×2 分块展成 token 序列 `[B, (H/16)*(W/16), 64]`。
2. **条件图 token 拼接（核心 trick）**：若 `edit_latents` 存在，同样 patchify 后**直接 concat 到序列末尾**形成 `[B, N_img + N_edit, 64]`，DiT 自注意力天然把条件信息融入目标 token。这就是当前"图像 unwarp"的主要机制——**不是 ControlNet、不是 cross-attn，而是序列拼接 + self-attention**。
3. **时序/文本嵌入**：`timestep` 走 `time_text_embed`；`prompt_emb` 走 `txt_norm → txt_in`；`pos_embed` 按 `img_shapes`（包含原图与条件图两段）生成 RoPE。
4. **DiT Blocks**：逐块 `transformer_block(image, text, temb, rotary_emb, ...)`；每块都用 `gradient_checkpoint_forward` 包裹以省显存；若启用 blockwise controlnet，会在每块之后对前 `N_img` 个 token 做残差加法（训练未启用）。
5. **裁回**：`norm_out → proj_out → [:, :image_seq_len]`（丢弃条件图 token 位置的输出），再 unpatchify 回 `[B, 16, H/8, W/8]`，作为 `noise_pred`。

### 0.4 使用到的核心方法/技术点汇总

| 方法 | 位置 | 作用 |
| --- | --- | --- |
| **Flow Matching（非 DDPM）** | [diffsynth/schedulers/flow_match.py](diffsynth/schedulers/flow_match.py) | `add_noise: x_t=(1-σ)x_0+σ·ε`；`training_target = ε - x_0`；时间步加权 `linear_timesteps_weights`（高斯钟形，均衡中间 timestep） |
| **Exponential Shift + shift_terminal** | [flow_match.py:44-52](diffsynth/schedulers/flow_match.py#L44-L52) | 把 σ 分布按序列长度自适应 shift（`calculate_shift`），让不同分辨率共享同一 scheduler |
| **LoRA on DiT** | [train.py:42-55](examples/qwen_image/model_training/train.py#L42-L55) + [diffsynth/lora/](diffsynth/lora/) | 只训练 12 类线性层的低秩旁路，rank=32；其他全部 `freeze_except` 冻结 |
| **Edit-token 序列拼接条件** | [qwen_image.py:628-632](diffsynth/pipelines/qwen_image.py#L628-L632) | 条件图 latent 与目标 latent 在 token 维 concat，由 self-attention 融合；最后裁回目标段 |
| **Qwen-Image-Edit 多模态文本条件** | [qwen_image.py:421-456](diffsynth/pipelines/qwen_image.py#L421-L456) | `Qwen2VLProcessor` 把 `edit_image` 作为 vision token 与文本一起过 text encoder，产生统一 `prompt_emb` |
| **VAE latent（16 ch, ×8 下采样）** | `QwenImageUnit_InputImageEmbedder` / `EditImageEmbedder` | 图像→latent，8× 下采样、16 通道，后续 DiT 再 2×2 patchify |
| **Gradient Checkpointing** | [qwen_image.py:651-661](diffsynth/pipelines/qwen_image.py#L651-L661) | 每个 transformer block 外包 `gradient_checkpoint_forward` |
| **Accelerate Multi-GPU + bf16** | [scripts/Acceconfig_8A800.yaml](scripts/Acceconfig_8A800.yaml) | 8 卡 DDP + bf16 自动混精；`find_unused_parameters=True` |
| **按需 VRAM 管理** | [qwen_image.py:99-220](diffsynth/pipelines/qwen_image.py#L99-L220) | 每个 unit 声明 `onload_model_names`，Runner 只把需要的模型拉到 GPU，训练脚本在 8×A800 默认常驻 |
| **AdamW + ConstantLR** | [train.py:130-131](examples/qwen_image/model_training/train.py#L130-L131) | `lr=1e-4, wd=0.01`，恒定学习率 6 epoch |

### 0.5 训练 vs 推理的差异

- **训练**：`pipe.scheduler.set_timesteps(1000, training=True)` 产生连续 1000 个 timestep 与训练权重；每步**只跑一次** forward，随机采一个 t，算 MSE。
- **推理**（[qwen_image_batch_full_lora.py](qwen_image_batch_full_lora.py) / `QwenImagePipeline.__call__`）：`set_timesteps(num_inference_steps=...)` 构造调度表，从 `x_T = noise` 开始**循环** `scheduler.step` 去噪；`cfg_scale != 1` 时跑一次 posi + 一次 nega 做 classifier-free guidance；最后 `vae.decode` 回到像素空间。

### 0.6 端到端一句话总结

> **在 Qwen-Image-Edit 的 DiT 上挂 rank=32 的 LoRA，把目标图 latent 与扰动图（edit_image）latent 沿 token 维拼接，并用带图像条件的 Qwen2VL 文本嵌入作为 text token，训练目标是 flow-matching 的 velocity (noise − x₀)，加权 MSE 作为 loss，8×A800 bf16 DDP 跑 6 epoch、每 4000 步保存一次 LoRA `.safetensors`，推理时通过 `pipe.load_lora(pipe.dit, ckpt)` 复用。**

---

## 1. 项目定位

- 本仓库 fork 自 **DiffSynth-Studio**，当前主线任务是对 **Qwen-Image-Edit** 做 LoRA 微调，实际方向为**图像 unwarp / 矫正**（由 [scripts/train_sample.sh](scripts/train_sample.sh) 中的 `train_id=20250929-1_1in10_w_unwarp` 与 `1in10_w_metadata.csv` 推断）。
- 尽管目录名包含 `genwarp`，代码中并没有显式的 novel view synthesis / 相机位姿模块；"warp" 信息是以成对数据（`image` ↔ `edit_image`）作为 extra input 送入编辑模型，由 DiT 通过条件注入学习。
- `examples/` 下还存在 Flux、WanVideo 等其他训练入口，当前活跃的只有 Qwen-Image 路径。

## 2. 目录速览（与训练相关）

| 路径 | 作用 |
| --- | --- |
| [examples/qwen_image/model_training/train.py](examples/qwen_image/model_training/train.py) | **主训练入口**（Qwen-Image-Edit LoRA） |
| [examples/qwen_image/model_training/train_re.py](examples/qwen_image/model_training/train_re.py) | 训练入口的变体版本 |
| [diffsynth/trainers/utils.py](diffsynth/trainers/utils.py) | `DiffusionTrainingModule` / `ImageDataset` / `ModelLogger` / `launch_training_task` / `qwen_image_parser` |
| [diffsynth/trainers/text_to_image.py](diffsynth/trainers/text_to_image.py) | T2I LoRA 训练模块 |
| [diffsynth/pipelines/qwen_image.py](diffsynth/pipelines/qwen_image.py) | `QwenImagePipeline` + `training_loss` |
| [diffsynth/schedulers/](diffsynth/schedulers/) | `FlowMatchScheduler` |
| [diffsynth/lora/](diffsynth/lora/) | `GeneralLoRALoader` |
| [scripts/train_sample.sh](scripts/train_sample.sh) | 启动脚本示例 |
| [scripts/Acceconfig_8A800.yaml](scripts/Acceconfig_8A800.yaml) | Accelerate 分布式配置（8 × A800, bf16） |
| [dataset.py](dataset.py) | 仅 4 行，用于从 ModelScope 拉取自生成数据集，非训练用 Dataset |
| [qwen_image_batch_full_lora.py](qwen_image_batch_full_lora.py) | 训练完成后的批量推理脚本 |
| [matrix_cal.py](matrix_cal.py) | GPU 显存测试工具，与训练无关 |

## 3. 启动方式

```bash
bash scripts/train_sample.sh
```

核心命令（简化自 [scripts/train_sample.sh](scripts/train_sample.sh)）：

```bash
accelerate launch [--config_file scripts/Acceconfig_8A800.yaml] \
    examples/qwen_image/model_training/train.py \
    --dataset_base_path <csv 所在目录> \
    --dataset_metadata_path <csv 路径> \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --max_pixels 1048576 \
    --dataset_repeat 5 \
    --learning_rate 1e-4 \
    --num_epochs 6 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path <checkpoint 输出目录> \
    --lora_base_model "dit" \
    --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --dataset_num_workers 2 \
    --find_unused_parameters \
    --save_steps 4000 \
    --tokenizer_path <tokenizer> --processor_path <processor> \
    --model_paths '[[DiT 9 shards], [TextEncoder 4 shards], VAE]'
```

分布式设置（来自 [scripts/Acceconfig_8A800.yaml](scripts/Acceconfig_8A800.yaml)）：

- `distributed_type: MULTI_GPU`
- `num_processes: 8`
- `mixed_precision: bf16`
- `main_process_port: 29555`
- `enable_cpu_affinity: true`

## 4. 数据流

- **元数据**：CSV，字段包含 `image` / `edit_image` 等（由 `--data_file_keys` 指定列），每行一条样本。
- **Dataset**：`ImageDataset`（[diffsynth/trainers/utils.py:131-148](diffsynth/trainers/utils.py#L131-L148)）
    - 支持 JSON / JSONL / CSV 或自动扫描元数据
    - `__getitem__` 返回形如 `{"prompt": str, "image": PIL.Image, "edit_image": PIL.Image, ...}`
    - 动态分辨率：由 `max_pixels=1048576` 与 VAE factor 共同决定 `crop_and_resize`，保证长宽能被 factor 整除
    - `dataset_repeat=5`：把 epoch 内的样本数逻辑性放大 5 倍
- **batch_size**：代码中未显式暴露 CLI，实际为 1（[train.py:132-133](examples/qwen_image/model_training/train.py#L132-L133) 有 TODO 注释），吞吐依赖 `gradient_accumulation_steps` 与多卡数据并行。

## 5. 模型结构

`QwenImagePipeline`（[diffsynth/pipelines/qwen_image.py](diffsynth/pipelines/qwen_image.py)）由以下组件组成：

| 组件 | 说明 | 训练状态 |
| --- | --- | --- |
| DiT (`pipe.dit`) | QwenImageDiT，9 shards | **注入 LoRA**（rank=32） |
| Text Encoder | 4 shards | 冻结 |
| VAE | 单文件 | 冻结 |
| Tokenizer / Processor | 来自 Qwen-Image-Edit | — |
| Scheduler | `FlowMatchScheduler`，`set_timesteps(1000, training=True)` | — |
| ControlNet / BlockwiseControlNet | 可选，通过 `--extra_inputs` 以 `controlnet_*` / `blockwise_controlnet_*` 前缀开启 | 当前脚本未启用 |

LoRA 注入的 12 个模块：
`to_q, to_k, to_v, add_q_proj, add_k_proj, add_v_proj, to_out.0, to_add_out, img_mlp.net.2, img_mod.1, txt_mlp.net.2, txt_mod.1`。

冻结策略：`self.pipe.freeze_except(...)` 只解冻 `trainable_models` 指定的模块，再把 LoRA 单独作为可训练参数注入。

## 6. 训练循环（Forward → Loss → Backward）

**入口：** [examples/qwen_image/model_training/train.py:111-141](examples/qwen_image/model_training/train.py#L111-L141)

1. 构造 `ImageDataset` 与 `QwenImageTrainingModule`
2. 优化器 / 学习率：`AdamW(lr=args.learning_rate, weight_decay=args.weight_decay)` + `ConstantLR`
3. 调用 `launch_training_task(...)`（[diffsynth/trainers/utils.py:420-454](diffsynth/trainers/utils.py#L420-L454)）进入主循环：

```
for epoch in range(num_epochs):
    for data in dataloader:
        optimizer.zero_grad()
        loss = model(data)                       # → forward()
        accelerator.backward(loss)
        optimizer.step()
        model_logger.on_step_end(...)            # 达到 save_steps 时保存 LoRA
        scheduler.step()
```

**forward**（[train.py:103-107](examples/qwen_image/model_training/train.py#L103-L107)）：

- `forward_preprocess(data)`：拼接 `prompt`、`input_image`、`extra_inputs`（如 `edit_image`），流经 `pipe.units` 做 VAE 编码、text 编码、时间步采样、ControlNet 注入等。
- `pipe.training_loss(**models, **inputs)` 计算最终 loss。

**Loss**：Flow-Matching 风格的加权 MSE（[diffsynth/pipelines/qwen_image.py:85-96](diffsynth/pipelines/qwen_image.py#L85-L96)）

```python
loss = F.mse_loss(noise_pred.float(), training_target.float())
loss = loss * self.scheduler.training_weight(timestep)
```

## 7. 显存与稳定性策略

- `use_gradient_checkpointing=True`：DiT 开启梯度检查点
- `mixed_precision=bf16`：Accelerate 统一到 bfloat16
- `find_unused_parameters=True`：DDP 容忍部分参数在前向中未参与梯度
- 可选 `use_gradient_checkpointing_offload` 把激活 offload 到 CPU（当前未启用）
- `enable_cpu_affinity: true`：绑核减少 NUMA 抖动

## 8. Checkpoint 机制

**实现**：[diffsynth/trainers/utils.py:387-417](diffsynth/trainers/utils.py#L387-L417) 的 `ModelLogger`

- **保存内容**：仅可训练权重（LoRA），通过 `export_trainable_state_dict` 提取
- **前缀处理**：`--remove_prefix_in_ckpt "pipe.dit."` 把 state_dict 前缀剥掉，方便推理侧直接 `pipe.load_lora(pipe.dit, ckpt)`
- **触发时机**：每 `save_steps=4000` 步存一次；若未设置则每 epoch 存一次；训练结束再补存一次
- **格式**：`.safetensors`
- **位置**：`--output_path` 指向的目录

## 9. 从断点恢复

- `--lora_checkpoint <path>`：在 `QwenImageTrainingModule.__init__` 中加载已训练 LoRA 作为起点（[train.py:48-54](examples/qwen_image/model_training/train.py#L48-L54)）。
- **注意**：当前实现不恢复 optimizer / scheduler 状态，严格意义上并非完整 resume，只是 LoRA 权重热启。

## 10. 训练后推理 / 验证

- **批量推理**：[qwen_image_batch_full_lora.py](qwen_image_batch_full_lora.py)，`pipe.load_lora(pipe.dit, lora_path)` 后做 image-to-image 生成；支持 crop / stretch / scale 三种 resize。
- **单样本验证**：
    - [examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit.py](examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit.py)
    - [examples/qwen_image/model_training/validate_full/Qwen-Image-Edit.py](examples/qwen_image/model_training/validate_full/Qwen-Image-Edit.py)
- **启动**：`bash scripts/test_sample.sh`

## 11. 典型问题自查清单

- DiT 9 个 shard 路径是否齐全（shell 脚本里写死的列表）
- `--data_file_keys` 与 `--extra_inputs` 是否匹配 CSV 字段
- `--max_pixels` 过大时注意显存（8×A800 80G 下 1M 像素 + bf16 + grad ckpt 正常）
- `find_unused_parameters=True` 会轻微降速，排障后可关
- `dataset_repeat` 会影响有效 epoch 步数，换数据集时重新评估
- batch_size 实际为 1，若想增大吞吐优先调 `gradient_accumulation_steps`

---

## 附：端到端调用链速查

```
scripts/train_sample.sh
  └── accelerate launch examples/qwen_image/model_training/train.py
        ├── qwen_image_parser()                 # diffsynth/trainers/utils.py
        ├── ImageDataset(args)                  # diffsynth/trainers/utils.py:131
        ├── QwenImageTrainingModule(...)        # train.py:10
        │     ├── QwenImagePipeline.from_pretrained(...)
        │     ├── pipe.freeze_except(...)
        │     └── add_lora_to_model(pipe.dit, ...)
        ├── AdamW + ConstantLR
        └── launch_training_task(...)           # diffsynth/trainers/utils.py:420
              └── for epoch, batch:
                    ├── model(data)             # → forward_preprocess → pipe.training_loss
                    │     └── F.mse_loss(noise_pred, target) * scheduler.training_weight(t)
                    ├── accelerator.backward(loss)
                    ├── optimizer.step()
                    ├── ModelLogger.on_step_end()    # 保存 LoRA safetensors
                    └── scheduler.step()
```
