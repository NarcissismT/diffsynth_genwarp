#!/usr/bin/env bash
# One-GPU, one-train-batch, one-validation-batch real-Qwen smoke test.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

CONFIG="${CONFIG:-configs/unified.yaml}"
RESUME="${RESUME:-runs/d2r/unified/resume_from_v2_epoch8.pt}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-9}"
MASTER_PORT="${MASTER_PORT:-29534}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/preflight_v31/smoke}"
PYTHON="${PYTHON:-/usr/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python)"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[smoke] preflight manifest/checkpoint/flow canvases"
"$PYTHON" scripts/preflight_v31.py \
    --config "$CONFIG" \
    --checkpoint "$RESUME" \
    --sample-count "${PREFLIGHT_SAMPLES:-2}" \
    --max-mae "${MAX_GT_MAE:-0.08}" \
    --output-dir runs/preflight_v31/checks

echo "[smoke] one real Qwen train batch + one validation batch"
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    -m diffusion2raft.train \
    --config "$CONFIG" \
    --stage unified \
    --resume "$RESUME" \
    --epochs "$SMOKE_EPOCHS" \
    --max-train-steps 1 \
    --max-val-batches 1 \
    --preview-every 1 \
    --output-dir "$OUTPUT_DIR"

echo "[smoke] passed"
printf -v SMOKE_EPOCH_PADDED '%04d' "$SMOKE_EPOCHS"
echo "[smoke] inspect: $OUTPUT_DIR/unified/previews/epoch_${SMOKE_EPOCH_PADDED}.jpg"
