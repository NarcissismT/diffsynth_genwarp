#!/usr/bin/env bash
# =============================================================================
# Unified 文档矫正模型训练（Stage A prior 已训好，这里只跑 unified 联合阶段）
#
# 与旧级联 joint 的区别：不生成/不缓存/不读取 Qwen RGB guide。Qwen 在模型内部
# 以 output_type=latent 抽取 transformer hidden tokens（不做 VAE decode），
# 联合优化 Stage-A prior + token projectors + 可靠性门控 + RAFT-like refiner。
# 最终 RGB 始终从 warped 原图 grid_sample 采样。
#
# 运行环境：pyxis/enroot 容器 registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers
#   注意：必须用 v2-diffusers（在 v2 基础上装好了 diffusers==0.35.2，系统级
#   /usr/bin/python 可 import QwenImageEditPipeline）。原始 v2 里没有 diffusers，
#   unified 阶段会 ModuleNotFoundError。SLURM 无法 pip install，故预装进镜像。
# 参考 train_v4_4_layer36_val.sh 的容器/宿主机 python 判别方式。
#
# 前置：
#   1. Stage-A prior checkpoint 已放在 runs/d2r/prior/latest.pt（本仓库已 staged，
#      来自 diffusion2raft_reference_implementation_v1 训好的 512 prior，EPE≈6.06）。
#   2. data/train.jsonl 与 data/val.jsonl 已生成（scripts/make_manifest_from_csv.py）。
#   3. configs/unified.yaml 里 qwen.model_id 指向本地 Qwen 目录、local_files_only: true。
# =============================================================================
set -euo pipefail

# ---- 可调参数 ----
CONFIG="configs/unified.yaml"
STAGE="unified"
RESUME="runs/d2r/prior/latest.pt"   # Stage-A prior -> unified 迁移
EPOCHS=60
MASTER_PORT="${MASTER_PORT:-29531}"

# ---- 切到仓库根目录（scripts/ 的上一级）----
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

# ---- 容器内用系统 python，宿主机回退到自带环境 ----
# pyxis/enroot 不创建 /.dockerenv，用 dist-packages/torch 是否存在来判别（同参考脚本）。
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    echo "[warn] 未检测到容器内 torch；回退到 miniconda base python，Qwen 依赖可能缺失。" >&2
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/bin/python"
fi

# ---- 让 diffusion2raft 可 import（不 pip install，避免与其它仓库同名包冲突）----
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
# 离线：禁止任何联网下载，强制本地权重。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- GPU 数：默认按可见 GPU 自动推断，可用 NUM_GPUS 覆盖 ----
if [ -n "${NUM_GPUS:-}" ]; then
    NPROC="$NUM_GPUS"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
else
    NPROC="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
fi
echo "[info] repo=$REPO_ROOT python=$PYTHON nproc_per_node=$NPROC"
echo "[info] unified 阶段要求每 rank data.batch_size=1（QwenImageEditPlus 单图约束），"
echo "       全局 batch = $NPROC。DistributedSampler / DDP / rank0 存盘已在 train.py 实现。"

# ---- 训练 ----
# 单卡时 torchrun 也能跑（会退化为非分布式；train.py 用 LOCAL_RANK 判别）。
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    -m diffusion2raft.train \
    --config "$CONFIG" \
    --stage "$STAGE" \
    --resume "$RESUME" \
    --epochs "$EPOCHS"

# =============================================================================
# 显存说明：Qwen transformer BF16 权重约 39GB，单张 A10(23GB) 装不下。
#   - 大显存卡(A100/H100 80GB)：直接 BF16 跑，配置默认即可。
#   - 显存不足：configs/unified.yaml 里设 qwen.feature_quantization: 4bit
#     （需容器内 bitsandbytes），或 qwen.cpu_offload: true（慢）。
#
# 观察项（train.py 已打印 / 验证归并）：
#   epe / prior_epe（unified 应低于 prior，否则先查 prior 是否完整加载）、
#   fold_rate（上升就把 max_residual_px 24->16、lr_unified 1e-4->5e-5）、
#   residual_p95、feature_confidence。
# =============================================================================
