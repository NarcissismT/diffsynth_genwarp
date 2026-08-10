#!/usr/bin/env bash
# One-command v3.3 anchor/best inference, validation, evaluation, and report run.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

select_python() {
    local candidate
    if [[ -n "${PYTHON:-}" ]]; then
        [[ -x "$PYTHON" ]] || {
            echo "[error] PYTHON 不可执行：$PYTHON" >&2
            exit 69
        }
        "$PYTHON" -c 'import cv2, numpy, torch, yaml' >/dev/null 2>&1 || {
            echo "[error] PYTHON 缺少 cv2/numpy/torch/yaml 依赖：$PYTHON" >&2
            exit 69
        }
        return
    fi
    for candidate in \
        "$REPO_ROOT/../../miniconda3/bin/python" \
        /usr/bin/python \
        "$(command -v python 2>/dev/null || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]] \
            && "$candidate" -c 'import cv2, numpy, torch, yaml' >/dev/null 2>&1; then
            PYTHON="$candidate"
            return
        fi
    done
    echo "[error] 找不到同时具备 cv2/numpy/torch/yaml 的 Python" >&2
    exit 69
}

select_python

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="${CONFIG:-configs/unified_v3_3_teacher_anchor.yaml}"
QUALITY_POLICY="${QUALITY_POLICY:-configs/typical_v33_quality_v2.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/d2r_v3_3_teacher_anchor}"
ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT:-${RUN_ROOT}/unified/anchor.pt}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-${RUN_ROOT}/unified/best.pt}"
LATEST_CHECKPOINT="${LATEST_CHECKPOINT:-${RUN_ROOT}/unified/latest.pt}"
INPUT_DIR="${INPUT_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical}"
TARGET_FIRST_DIR="${TARGET_FIRST_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/test_silver_bullet_imgs/typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999}"
TARGET_SECOND_DIR="${TARGET_SECOND_DIR:-${TARGET_FIRST_DIR}-2nd}"
EXPECTED_COUNT="${EXPECTED_COUNT:-40}"

arguments=(
    --config "$CONFIG"
    --quality-policy "$QUALITY_POLICY"
    --run-root "$RUN_ROOT"
    --anchor-checkpoint "$ANCHOR_CHECKPOINT"
    --best-checkpoint "$BEST_CHECKPOINT"
    --latest-checkpoint "$LATEST_CHECKPOINT"
    --input-dir "$INPUT_DIR"
    --target-first-dir "$TARGET_FIRST_DIR"
    --target-second-dir "$TARGET_SECOND_DIR"
    --run-id "$RUN_ID"
    --expected-count "$EXPECTED_COUNT"
)

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    arguments+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "${REPORT_DIR:-}" ]]; then
    arguments+=(--report-dir "$REPORT_DIR")
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    arguments+=(--preflight-only)
fi
if [[ "${SKIP_INFERENCE:-0}" == "1" ]]; then
    arguments+=(--skip-inference)
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" scripts/finalize_typical_v33.py "${arguments[@]}"
