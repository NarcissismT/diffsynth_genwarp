#!/usr/bin/env bash
# Start the safe v3.2 continuation detached from the terminal.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

export CONFIG="${CONFIG:-configs/unified_v3_2.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_2}"
export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"
export SEED_RESUME_REASON="${SEED_RESUME_REASON:-v3.1 seed}"
export EPOCHS="${EPOCHS:-32}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-v3.2}"
export TRAIN_LOG_PREFIX="${TRAIN_LOG_PREFIX:-train_v32}"
export PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-runs/preflight_v32/main}"
export MIN_RESUME_COMPLETED_EPOCHS=20
export ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"

exec "$REPO_ROOT/scripts/run_unified_v31_background.sh"
