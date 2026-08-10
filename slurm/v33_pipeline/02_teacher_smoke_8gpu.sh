#!/usr/bin/env bash
# Slurm job body: exactly 8 GPUs, at least 40 CPU, 3 hours recommended.
# Run only after 01_teacher_capacity_1gpu.sh exits successfully.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

d2r_cd_project
d2r_require_visible_gpus 8
d2r_verify_seed
d2r_require_capacity_pointer
d2r_acquire_job_lock production_pipeline
[[ ! -e "$D2R_SMOKE_REPORT_ROOT" && ! -L "$D2R_SMOKE_REPORT_ROOT" ]] \
    || d2r_fail "正式 smoke 输出目录已存在；请先保留并改名旧失败目录：$D2R_SMOKE_REPORT_ROOT"

echo "[info] 开始正式 8-GPU v3.3 smoke：report=$D2R_SMOKE_REPORT_ROOT"
env \
    -u D2R_TEACHER_CAPACITY_RECEIPT_B64 \
    PYTHON="$D2R_PYTHON" \
    CONFIG="$D2R_CONFIG" \
    SEED_CHECKPOINT="$D2R_SEED_CHECKPOINT" \
    TEACHER_CAPACITY_POINTER="$D2R_CAPACITY_POINTER" \
    SMOKE_NPROC=8 \
    FAILURE_WORLD_SIZES="2 8" \
    MIN_SEED_COMPLETED_EPOCHS=20 \
    ALLOW_INCOMPLETE_SEED=0 \
    ALLOW_PARTIAL_SMOKE=0 \
    MASTER_PORT=29633 \
    FAILURE_TIMEOUT_SECONDS=120 \
    FUNCTIONAL_TIMEOUT_SECONDS=3600 \
    RUN_ID="$D2R_SMOKE_RUN_ID" \
    REPORT_ROOT="$D2R_SMOKE_REPORT_ROOT" \
    bash scripts/smoke_unified_v33_teacher.sh

"$D2R_PYTHON" scripts/verify_v33_formal_smoke_report.py \
    --overall-report "$D2R_SMOKE_OVERALL_REPORT" \
    --expected-seed "$D2R_SEED_CHECKPOINT" \
    --expected-seed-sha256 "$D2R_EXPECTED_SEED_SHA256" \
    --expected-config "$D2R_CONFIG" \
    --expected-teacher-sha256 "$D2R_EXPECTED_TEACHER_SHA256"
echo "D2R_V33_FORMAL_SMOKE_JOB_PASS report=$D2R_SMOKE_OVERALL_REPORT"
