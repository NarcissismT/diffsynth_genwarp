#!/usr/bin/env bash
# Safe foreground entrypoint for the v3.2 continuation.
# The implementation stays in train_unified_v3.sh so v3.1 and v3.2 share the
# same CUDA-first guard, checkpoint selection, validation, and writer lock.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

export CONFIG="${CONFIG:-configs/unified_v3_2.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_2}"
export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"
export SEED_RESUME_REASON="${SEED_RESUME_REASON:-v3.1 seed}"
export EPOCHS="${EPOCHS:-32}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-v3.2}"
export PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-runs/preflight_v32/main}"
# Do not make the safety floor environment-overridable. The named ALLOW flag
# below is the only supported escape hatch and always emits a warning.
export MIN_RESUME_COMPLETED_EPOCHS=20
export ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"

exec "$REPO_ROOT/scripts/train_unified_v3.sh"
