#!/usr/bin/env bash
# =============================================================================
# Unified v3 续训脚本（从 v2 unified checkpoint 续训，加入残差真值/correspondence/
# gate 校准等新损失）。对应 diffusion2raft_unified_v3。
#
# 与 v2 train_unified.sh 的区别：
#   - 从 v2 训到 epoch 2 的 unified checkpoint 续训（参数键完全兼容，已验证）
#   - 新损失项：residual_flow(残差真值) + qwen_match(correspondence) + gate 校准
#   - 新指标：gain / final_win_rate / residual_epe / r_valid / qwen_match_epe /
#            q_acc1 / q_adv / q_win / gate / gate_target
#   - feature_type=hidden（续训不要中途切 qk）
#   - work_size=512（与 prior/v2 一致，切勿用 768）
#
# 运行环境：容器 registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers
#   （已预装 diffusers==0.35.2）。原始 v2 镜像没有 diffusers，会 ModuleNotFoundError。
#   SLURM 提交时容器镜像务必选 v2-diffusers。
#
# 前置（本仓库已就绪）：
#   - runs/d2r/unified/resume_from_v2_epoch2.pt  （v2 续训源，已 staged）
#   - data/train.jsonl (115716) / data/val.jsonl (300)  （已从 v2 复用）
#   - configs/unified.yaml：work_size=512、qwen 指向本地模型、feature_type=hidden
# =============================================================================
set -euo pipefail

# ---- 可调参数 ----
CONFIG="configs/unified.yaml"
STAGE="unified"
RESUME="runs/d2r/unified/resume_from_v2_epoch8.pt"   # v2 unified epoch2
EPOCHS=20            # 总训练到 epoch 20（不是再训 20 个）；续训从 epoch 3 起
MASTER_PORT="${MASTER_PORT:-29533}"

# ---- 切到仓库根目录 ----
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

# ---- 容器内用系统 python（判别方式同 v2 脚本）----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    echo "[warn] 未检测到容器内 torch；确认用的是 diffsynth:v2-diffusers 镜像。" >&2
    PYTHON="/usr/bin/python"
fi

# ---- import 路径 + 离线 ----
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- GPU 数：默认按可见 GPU 推断，可用 NUM_GPUS 覆盖 ----
if [ -n "${NUM_GPUS:-}" ]; then
    NPROC="$NUM_GPUS"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
else
    NPROC="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
fi
echo "[info] repo=$REPO_ROOT python=$PYTHON nproc_per_node=$NPROC"
echo "[info] 从 $RESUME 续训到 epoch $EPOCHS（unified 每 rank batch=1，全局 batch=$NPROC）。"
echo "[info] 新 loss 让 total 与旧日志不可直接比；应看 seq/epe/gain/final_win_rate。"

# ---- 续训 ----
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    -m diffusion2raft.train \
    --config "$CONFIG" \
    --stage "$STAGE" \
    --resume "$RESUME" \
    --epochs "$EPOCHS"

# =============================================================================
# 观察到 epoch 6（作者建议）：
#   q_adv > 0 且 q_win > 0.5   ：Qwen 匹配确实优于零残差回退 → 可继续训
#   gate 逐渐接近 gate_target   ：门控被正确校准（不要求一定升高）
#   r_valid > 0.9               ：残差真值有效区占比健康
#   gain 继续扩大、final_win_rate 稳定 > 0.5 ：unified 持续超越 prior
#
# 若 q_adv 始终不为正 → 从 Stage-A checkpoint 单独启 feature_type:qk 消融，
#   不要在当前 hidden checkpoint 中途切 qk（projections 通道维度不匹配）。
# 若 fold_rate 上升 → 先降 max_residual_px 24→16、lr_unified→5e-5。
# =============================================================================
