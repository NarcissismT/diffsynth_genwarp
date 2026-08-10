#!/usr/bin/env bash
# Controlled epoch-9 protocol sweep.  Submit this on a single GPU in the same
# diffsynth:v2-diffusers image used by training.  The default staged plan has
# eight cells: 3 temperatures x (both/none), plus matching-only/context-only at
# the checkpoint temperature.  Each cell is atomically persisted for resume.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

PYTHON="${PYTHON:-/usr/bin/python}"
CHECKPOINT="${CHECKPOINT:-runs/d2r_v3_1/unified/epoch_0009.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/d2r_v3_1/ablation_epoch0009_corr_qwen_residual.json}"
DEVICE="${DEVICE:-cuda:0}"
MAX_BATCHES="${MAX_BATCHES:-}"
TEMPERATURES="${TEMPERATURES:-9.797959 3.1301691647577132 1.0}"
QWEN_MODES="${QWEN_MODES:-both none}"
TRAINING_TEMPERATURE_QWEN_MODES="${TRAINING_TEMPERATURE_QWEN_MODES:-matching_only context_only}"
RESIDUAL_SCALES="${RESIDUAL_SCALES:-0 0.10 0.25 0.50 0.75 1.0}"
RESUME="${RESUME:-1}"

[[ -x "$PYTHON" ]] || { echo "[error] Python 不可执行：$PYTHON" >&2; exit 70; }
[[ -s "$CHECKPOINT" ]] || { echo "[error] 固定 checkpoint 不存在：$CHECKPOINT" >&2; exit 66; }

read -r -a temperature_args <<<"$TEMPERATURES"
read -r -a qwen_mode_args <<<"$QWEN_MODES"
read -r -a training_temperature_qwen_mode_args <<<"$TRAINING_TEMPERATURE_QWEN_MODES"
read -r -a residual_scale_args <<<"$RESIDUAL_SCALES"
extra_args=()
if [[ -n "$MAX_BATCHES" ]]; then
    [[ "$MAX_BATCHES" =~ ^[1-9][0-9]*$ ]] \
        || { echo "[error] MAX_BATCHES 必须是正整数" >&2; exit 64; }
    extra_args+=(--max-batches "$MAX_BATCHES")
fi
case "$RESUME" in
    1|true|TRUE|yes|YES) extra_args+=(--resume) ;;
    0|false|FALSE|no|NO) ;;
    *) echo "[error] RESUME 必须是 0/1 或 true/false" >&2; exit 64 ;;
esac

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -m diffusion2raft.ablate_residual_qwen \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --temperatures "${temperature_args[@]}" \
    --qwen-modes "${qwen_mode_args[@]}" \
    --training-temperature-qwen-modes "${training_temperature_qwen_mode_args[@]}" \
    --residual-scales "${residual_scale_args[@]}" \
    --output-json "$OUTPUT_JSON" \
    "${extra_args[@]}"
