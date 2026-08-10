#!/usr/bin/env bash
# Safe foreground entrypoint for the v3.3 teacher-anchored continuation.
# The shared launcher performs CUDA and checkpoint validation before it creates
# any lock, PID, status, or log state.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

export CONFIG="${CONFIG:-configs/unified_v3_3_teacher_anchor.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_3_teacher_anchor}"
export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"
export SEED_RESUME_REASON="${SEED_RESUME_REASON:-completed v3.1 seed}"
export EPOCHS="${EPOCHS:-32}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-v3.3 teacher-anchor}"
export PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-runs/preflight_v33_teacher_anchor/main}"
export REQUIRE_TEACHER_CAPACITY_EVIDENCE=1
export TEACHER_CAPACITY_POINTER="${TEACHER_CAPACITY_POINTER:-runs/preflight_v33_teacher_capacity/approved.json}"
# The safety floor is fixed. ALLOW_RESUME_BELOW_MIN_EPOCHS=1 is the only
# supported, warning-emitting escape hatch.
export MIN_RESUME_COMPLETED_EPOCHS=20
export ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"
# The production teacher trace contains literal cuda:0 allocations.  Keep all
# physical GPUs for DDP, but expose exactly one as logical cuda:0 per worker.
export ISOLATE_CUDA_PER_RANK=1

exec "$REPO_ROOT/scripts/train_unified_v3.sh"
