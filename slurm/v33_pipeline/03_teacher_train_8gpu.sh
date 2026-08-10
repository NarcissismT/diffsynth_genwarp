#!/usr/bin/env bash
# Slurm job body: exactly 8 GPUs, at least 40 CPU, about 120 hours total.
# Safe to resubmit after a time limit: the v3.3 launcher resumes its own latest.pt.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

d2r_cd_project
d2r_require_visible_gpus 8
d2r_verify_seed
d2r_require_capacity_pointer
"$D2R_PYTHON" scripts/verify_v33_formal_smoke_report.py \
    --overall-report "$D2R_SMOKE_OVERALL_REPORT" \
    --expected-seed "$D2R_SEED_CHECKPOINT" \
    --expected-seed-sha256 "$D2R_EXPECTED_SEED_SHA256" \
    --expected-config "$D2R_CONFIG" \
    --expected-teacher-sha256 "$D2R_EXPECTED_TEACHER_SHA256"
d2r_acquire_job_lock production_pipeline

echo "[info] 开始/恢复正式 v3.3 teacher-anchor 训练：target_completed_epochs=32"
env \
    -u RESUME \
    -u ALLOW_RESUME_OVERRIDE \
    -u OUTPUT_STAGE_DIR \
    -u LATEST_CHECKPOINT \
    -u STATE_DIR \
    -u LOCK_FILE \
    -u PID_FILE \
    -u STATUS_FILE \
    -u TRAIN_LOCK_FD \
    -u TRAIN_LOG_FILE \
    -u D2R_TEACHER_CAPACITY_RECEIPT_B64 \
    PYTHON="$D2R_PYTHON" \
    CONFIG="$D2R_CONFIG" \
    OUTPUT_ROOT="$D2R_V33_OUTPUT_ROOT" \
    STAGE=unified \
    EPOCHS=32 \
    NUM_GPUS=8 \
    SEED_RESUME="$D2R_SEED_CHECKPOINT" \
    TEACHER_CAPACITY_POINTER="$D2R_CAPACITY_POINTER" \
    RUN_PREFLIGHT=1 \
    PREFLIGHT_OUTPUT_DIR=runs/preflight_v33_teacher_anchor/main \
    PREFLIGHT_SAMPLES=3 \
    MAX_GT_MAE=0.08 \
    MASTER_PORT=29643 \
    BEST_METRIC=line_epe \
    CHECK_GPU_ONLY=0 \
    CHECK_LAUNCH_ONLY=0 \
    TRAIN_LOCK_HELD=0 \
    ALLOW_RESUME_BELOW_MIN_EPOCHS=0 \
    bash scripts/train_unified_v33_teacher.sh

final_checkpoint="$D2R_V33_OUTPUT_ROOT/unified/epoch_0032.pt"
latest_checkpoint="$D2R_V33_OUTPUT_ROOT/unified/latest.pt"
anchor_checkpoint="$D2R_V33_OUTPUT_ROOT/unified/anchor.pt"
status_file="$D2R_V33_OUTPUT_ROOT/unified/.launcher/last_exit.status"
pid_file="$D2R_V33_OUTPUT_ROOT/unified/.launcher/train.pid"

[[ -f "$anchor_checkpoint" && -s "$anchor_checkpoint" && ! -L "$anchor_checkpoint" ]] \
    || d2r_fail "训练结束但 immutable anchor.pt 缺失"
d2r_checkpoint_status "$anchor_checkpoint" 20 21
d2r_checkpoint_status "$latest_checkpoint" 31 32
d2r_checkpoint_status "$final_checkpoint" 31 32
[[ -f "$status_file" && ! -L "$status_file" ]] \
    || d2r_fail "训练 launcher 状态文件缺失"
grep -qx 'exit_code=0' "$status_file" \
    || d2r_fail "训练 launcher 未记录 exit_code=0"
[[ ! -e "$pid_file" && ! -L "$pid_file" ]] \
    || d2r_fail "训练结束后仍残留 active train.pid"

final_receipt="$(
    env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
        "$D2R_PYTHON" scripts/teacher_capacity_production.py verify \
        --config "$D2R_CONFIG" \
        --pointer "$D2R_CAPACITY_POINTER" \
        --resume "$final_checkpoint"
)" || d2r_fail "最终 teacher checkpoint 的 capacity receipt 复验失败"
d2r_verify_receipt_text "$final_receipt"
latest_receipt="$(
    env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
        "$D2R_PYTHON" scripts/teacher_capacity_production.py verify \
        --config "$D2R_CONFIG" \
        --pointer "$D2R_CAPACITY_POINTER" \
        --resume "$latest_checkpoint"
)" || d2r_fail "latest.pt 的 capacity receipt 复验失败"
d2r_verify_receipt_text "$latest_receipt"
anchor_receipt="$(
    env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
        "$D2R_PYTHON" scripts/teacher_capacity_production.py verify \
        --config "$D2R_CONFIG" \
        --pointer "$D2R_CAPACITY_POINTER" \
        --resume "$anchor_checkpoint"
)" || d2r_fail "immutable anchor.pt 的 capacity receipt 复验失败"
d2r_verify_receipt_text "$anchor_receipt"
[[ "$anchor_receipt" == "$final_receipt" \
    && "$latest_receipt" == "$final_receipt" ]] \
    || d2r_fail "anchor/latest/final checkpoint 保存的 capacity receipt 不一致"
echo "D2R_V33_TRAIN_COMPLETE checkpoint=$final_checkpoint"
