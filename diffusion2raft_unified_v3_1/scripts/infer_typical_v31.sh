#!/usr/bin/env bash
# Run the fixed 40-image real-world set with one Qwen/model load.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"
PYTHON="${PYTHON:-/usr/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python)"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="${CHECKPOINT:-runs/d2r_v3_1/unified/best.pt}"
INPUT_DIR="${INPUT_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/d2r_v3_1/typical_best_letterbox}"

echo "[info] checkpoint=$CHECKPOINT"
echo "[info] input=$INPUT_DIR output=$OUTPUT_DIR"
"$PYTHON" -m diffusion2raft.infer \
    --config configs/unified.yaml \
    --checkpoint "$CHECKPOINT" \
    --stage unified \
    --warped-dir "$INPUT_DIR" \
    --glob '*.jpg' \
    --resize-policy letterbox \
    --padding-mode replicate \
    --output-dir "$OUTPUT_DIR"

