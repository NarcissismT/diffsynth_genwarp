#!/usr/bin/env bash
# Run one strictly bound v3.3 teacher checkpoint on the fixed 40-image set.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"
PYTHON="${PYTHON:-/usr/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python)"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-configs/unified_v3_3_teacher_anchor.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/d2r_v3_3_teacher_anchor/unified/anchor.pt}"
INPUT_DIR="${INPUT_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/d2r_v3_3_teacher_anchor/typical_anchor_corrected512}"

expected_checkpoint_arguments=()
if [[ -n "${EXPECTED_CHECKPOINT_PATH:-}" \
   || -n "${EXPECTED_CHECKPOINT_SIZE_BYTES:-}" \
   || -n "${EXPECTED_CHECKPOINT_MTIME_NS:-}" \
   || -n "${EXPECTED_CHECKPOINT_SHA256:-}" ]]; then
    for variable_name in \
        EXPECTED_CHECKPOINT_PATH \
        EXPECTED_CHECKPOINT_SIZE_BYTES \
        EXPECTED_CHECKPOINT_MTIME_NS \
        EXPECTED_CHECKPOINT_SHA256; do
        if [[ -z "${!variable_name:-}" ]]; then
            echo "[error] checkpoint provenance 参数必须完整，缺少：${variable_name}" >&2
            exit 64
        fi
    done
    expected_checkpoint_arguments=(
        --expected-checkpoint-path "$EXPECTED_CHECKPOINT_PATH"
        --expected-checkpoint-size-bytes "$EXPECTED_CHECKPOINT_SIZE_BYTES"
        --expected-checkpoint-mtime-ns "$EXPECTED_CHECKPOINT_MTIME_NS"
        --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
    )
fi

if [[ ! -f "$CHECKPOINT" || ! -s "$CHECKPOINT" ]]; then
    echo "[error] checkpoint 不存在或为空：$CHECKPOINT" >&2
    exit 66
fi

"$PYTHON" scripts/checkpoint_status.py \
    --checkpoint "$CHECKPOINT" \
    --expect-stage unified \
    --format human

echo "[info] config=$CONFIG"
echo "[info] checkpoint=$CHECKPOINT"
echo "[info] input=$INPUT_DIR output=$OUTPUT_DIR"
"$PYTHON" -m diffusion2raft.infer \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    "${expected_checkpoint_arguments[@]}" \
    --stage unified \
    --warped-dir "$INPUT_DIR" \
    --glob '*.jpg' \
    --output-dir "$OUTPUT_DIR"
