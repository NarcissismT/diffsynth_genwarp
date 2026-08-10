#!/usr/bin/env bash
# =============================================================================
# Qwen-off 消融：用同一个训练好的 checkpoint，在验证集上跑两遍——
#   qwen_on : 正常前向
#   qwen_off: 门控强制 0，reference/source 退化成纯 CNN fallback（完全绕开 Qwen）
# 打印两者的 EPE / line_epe / epe_gain 等对比。该实验只判断同一 checkpoint
# 中 Qwen 两条路径的净贡献，不构成移除 Qwen 的建议；路径拆分和 residual
# 强度应使用 ablate_residual_qwen_sweep.sh。
#
# 单卡即可（加载一次 39GB Qwen，不用 DDP）。容器：diffsynth:v2-diffusers。
# =============================================================================
set -euo pipefail

CONFIG="configs/unified.yaml"
# 用当前训练的 best/latest checkpoint。改这里切换要评的 checkpoint。
# 注意：SLURM 直接 execve 不支持行内 `VAR=x bash ...` 赋值前缀，所以在脚本里配。
CHECKPOINT="runs/d2r_v3_1/unified/epoch_0009.pt"
OUTPUT_JSON="runs/d2r_v3_1/qwen_off_ablation_epoch0009_correct_temp.json"
MAX_BATCHES=""   # 留空=全部 300 张验证；填数字(如 30)可快速抽查

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    echo "[warn] 未检测到容器内 torch；确认用的是 diffsynth:v2-diffusers 镜像。" >&2
    PYTHON="/usr/bin/python"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 固定 epoch snapshot，避免 best.pt 在长时间评估期间被训练任务覆盖。
if [ ! -f "$CHECKPOINT" ]; then
    echo "[error] 固定消融 checkpoint 不存在：$CHECKPOINT" >&2
    exit 66
fi
echo "[info] checkpoint=$CHECKPOINT  (单卡 Qwen-off 消融)"

extra=""
[ -n "$MAX_BATCHES" ] && extra="--max-batches $MAX_BATCHES"

"$PYTHON" -m diffusion2raft.ablate_qwen \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-json "$OUTPUT_JSON" \
    $extra

# =============================================================================
# 读法：
# 若 off 不差于 on，继续看 matching/context 拆分，不直接删除 Qwen；若 on 更好，
# 保留 Qwen 并定位收益路径。
# =============================================================================
