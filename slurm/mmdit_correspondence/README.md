# MMDiT correspondence experiment 1

本目录保留 base zero-shot 与 step-668000 LoRA 两个相互隔离的 8 卡实验结果，
不能混用结果目录或科学结论。
下列命令应在 Slurm 平台已经分配 8 张高显存数据中心 GPU 后执行；不要在没有 GPU allocation
的登录节点直接执行。所有实验参数都位于
`container_jobs/00_run_exp1_8xA800.sh`，外层 `srun` 只传三个缓存环境变量。
该入口会显式标记已经进入容器，避免嵌套 `srun`。

## 1. 当前默认 Exp1b：step-668000 LoRA 对照实验

按当前实验选择，`container_jobs/00_run_exp1_8xA800.sh` 使用：

- `/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit`
- `QwenImageEditPipeline`
- BF16、512×512、完整 50 steps、`output_type=latent`
- 加载 `/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors`
- LoRA rank=32、alpha=32、缩放为 1；启动前严格检查 SHA256、1440 个张量和 12 类 target modules
- 8 个独立单卡 worker 按样本分片，不做梯度/DDP 通信

这个本地 payload 正是 `scripts/f-20250929-1-train.sh` 列出的 9 个 Transformer
分片、4 个 text-encoder 分片、VAE、tokenizer 和 processor。入口会校验两个
shard index 引用的全部权重文件，然后按完全离线方式加载，不访问 Hugging Face。

该运行被单独标记为 `lora_ablation`，输出到
`/juicefs-algorithm/data/IPT/zhuochu_yang/mmdit_correspondence/runs/mmdit_correspondence_exp1b_step668000_lora`。
它测量的是文档矫正 LoRA 微调后的冻结特征对应性，不能标成 zero-shot 或 2511 主实验。
入口还会把 sanity/discovery/confirmation 三个 JSONL 与已经完成的 base zero-shot
参考目录逐字节比较，确保只改变 LoRA、样本与实验协议保持一致：
`/juicefs-algorithm/data/IPT/zhuochu_yang/mmdit_correspondence/runs/mmdit_correspondence_exp1_legacy_base_zero_shot`。

入口默认读取本地 CPU 解析渲染得到的正式 validation Manifest：
`artifacts/mmdit_correspondence/analytic_seed1337_512/manifests/validation.jsonl`。
首次实验前在仓库根目录运行
`bash scripts/prepare_mmdit_manifest_cpu.sh`；它固定选择 450 个文档，并按文档确定性
均衡轮换 light/medium/heavy 三档形变，
并验证 validation 中至少有 full profile 所需的 328 个独立文档，不需要 GPU。
生成后，IntSig Slurm portal 的提交内容固定为：

```bash
#!/bin/bash
# default bash

export HF_HOME=/juicefs-algorithm/data/IPT/yuang_feng/cache
export TRITON_CACHE_DIR=/tmp/slurm_${SLURM_JOB_ID}/triton
export TORCH_EXTENSIONS_DIR=/tmp/slurm_${SLURM_JOB_ID}/deepspeed_cache

srun --cpus-per-task 16 -K \
  --container-image=docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers \
  --container-mounts=/juicefs-algorithm:/juicefs-algorithm \
  --container-workdir=/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp \
  --container-env=HF_HOME,TRITON_CACHE_DIR,TORCH_EXTENSIONS_DIR \
  bash slurm/mmdit_correspondence/container_jobs/00_run_exp1_8xA800.sh
```

当前默认入口不需要下载 2511，也不需要运行
`scripts/prepare_mmdit_2511_offline_cache.sh`。外层 `srun` 命令无需改变。

直接处在已配置好 Python/Diffusers 的 8 卡容器内时，也可以使用较短的入口：

```bash
cd /path/to/diffsynth_genwarp
mkdir -p slurm/logs

MMDIT_MANIFEST=/absolute/path/validation_or_test.jsonl \
MMDIT_MANIFEST_ROLE=validation \
MMDIT_PROFILE=full \
MMDIT_EXPERIMENT_MODE=lora_ablation \
MMDIT_PIPELINE_CLASS=QwenImageEditPipeline \
MMDIT_MODEL_ID=/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit \
MMDIT_LORA_CHECKPOINT=/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors \
MMDIT_LORA_ALPHA=32 \
MMDIT_REFERENCE_RUN=/juicefs-algorithm/data/IPT/zhuochu_yang/mmdit_correspondence/runs/mmdit_correspondence_exp1_legacy_base_zero_shot \
bash slurm/mmdit_correspondence/run_exp1_8xA800.sh
```

## 2. 已完成的 base zero-shot 参考实验

base 参考实验使用同一份本地 `Qwen-Image-Edit` 与相同完整 split，但不加载 LoRA，
实验模式为 `legacy_base_zero_shot`。其已完成结果是本次 LoRA 实验的固定对照，
不应覆盖或复用其运行目录。

兼容入口 `run_exp1b_lora_8xA800.sh` 仍然保留，但 Slurm portal 应统一使用第 1 节的
`container_jobs/00_run_exp1_8xA800.sh`；所有参数都已经冻结在容器内脚本中。

## Manifest 契约

输入必须是 validation/test JSONL。至少需要：

```json
{
  "sample_id": "page_0001",
  "document_id": "document_001",
  "warped_image": "/path/warped.png",
  "rectified_image": "/path/rectified.png",
  "backward_map": "/path/backward_map.npy",
  "valid_mask": "/path/valid.npy",
  "input_size": [1024, 1024],
  "output_size": [1024, 1024],
  "warp_severity": "hard",
  "split": "validation"
}
```

`backward_map` 为 `[H,W,2]` 或 `[2,H,W]`，值是 warped source 原生图像中的绝对 `(x,y)` 像素坐标。`valid_mask` 可省略。`horizontal_structure`、`vertical_structure`、`boundary_structure` 若提供则必须一起提供，否则评估器使用并明确标记伪边缘区域。

full 需要至少 328 个唯一 `document_id`（8+64+256），pilot 需要至少 168 个（8+32+128）。三个阶段按文档互斥；pilot 只能作为预实验。

## 阶段与断点续跑

默认一次 allocation 顺序执行：

1. Sanity + deterministic repeat + document-disjoint batch shuffle；
2. Discovery：全 block、11 个 step、pre/post-RoPE 和 4 个温度；
3. Confirmation：独立样本、最多 6 个候选、完整 target lattice；
4. Seed repeat：Confirmation 前三、32 个样本、seed 0/1/2；
5. Final report。

Sanity Gate 强制要求：50 步、单 conditional forward、确定性复现、有限指标，以及最佳配置相对 shuffle 同时满足绝对下降 `>=0.05` 和 shuffle/normal 比例 `<=0.5`。后续阶段无法绕过上游门禁。

断点续跑示例：

```bash
MMDIT_STAGES=discovery,confirmation,seed_stability,final \
MMDIT_RUN_DIR=/absolute/path/existing_run \
MMDIT_MANIFEST=/absolute/path/same_manifest.jsonl \
MMDIT_MODEL_ID=/absolute/path/same_snapshot \
bash slurm/mmdit_correspondence/run_exp1_8xA800.sh
```

冻结配置会核对实验模式、模型、revision、pipeline、LoRA path/SHA 和 alpha。任何变化都应使用新的 `MMDIT_RUN_DIR`。

## 容器与预检

默认容器为：

```text
docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers
```

可通过 `MMDIT_CONTAINER_IMAGE` 覆盖。当前本地模型使用
`QwenImageEditPipeline`，由 `v2-diffusers` 镜像自带的 Diffusers 0.35.2 加载，
不需要安装 Diffusers 0.39，也不需要模型下载。GPU 入口强制离线，并逐项预检
Torch、Diffusers、Transformers、PEFT、Safetensors、PyArrow 和 Matplotlib。

脚本会在加载 8 份模型前运行全部单测；容器有 Pytest 时使用 Pytest，否则使用
仓库内的标准库测试 runner。仅在有意调试容器问题时才用 `MMDIT_RUN_TESTS=0`
显式关闭。

脚本检查：

- 正好 8 张可见 CUDA GPU；
- 每卡 CUDA compute capability 至少 8.0（支持 A800、H800、A100、H100 等）；
- 每卡显存至少约 75 GiB，因此 A100-40G 等小显存型号仍会被拒绝；
- 模型/pipeline/LoRA 与实验模式一致；
- 同一运行目录只有一个写入作业。

输出包括 rank-owned JSONL/Parquet、运行时指纹、scheduler trace、Discovery 热力图、Confirmation 分组指标、seed 稳定性、定性可视化和最终 Markdown 报告。Q/K 不落盘；EPE median/P95 使用可跨 rank 合并的 token-micro quantile sketch，并另报告 sample-macro 汇总。
