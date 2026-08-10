#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 1 hour recommended.
# This evaluates a candidate teacher but cannot approve or unlock production.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_CANDIDATE_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_CANDIDATE_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_CANDIDATE_REPORT="runs/preflight_v33_teacher_capacity/candidate-399999-full.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_diagnostic

[[ -f "$D2R_CANDIDATE_TEACHER" && -s "$D2R_CANDIDATE_TEACHER" \
    && ! -L "$D2R_CANDIDATE_TEACHER" ]] \
    || d2r_fail "399999 candidate 不存在、为空或为符号链接：$D2R_CANDIDATE_TEACHER"
candidate_digest_line="$(sha256sum -- "$D2R_CANDIDATE_TEACHER")" \
    || d2r_fail "无法计算 399999 candidate SHA-256"
candidate_sha="${candidate_digest_line%% *}"
[[ "$candidate_sha" == "$D2R_CANDIDATE_SHA256" ]] \
    || d2r_fail "399999 candidate SHA-256 已变化：expected=$D2R_CANDIDATE_SHA256 actual=$candidate_sha"
[[ ! -e "$D2R_CANDIDATE_REPORT" ]] \
    || d2r_fail "诊断报告已存在；请先改名保存后再重跑：$D2R_CANDIDATE_REPORT"

mkdir -p "$(dirname -- "$D2R_CANDIDATE_REPORT")"
echo "[info] 开始 diagnostic-only 399999 teacher capacity A/B"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/preflight_teacher_capacity.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_CANDIDATE_TEACHER" \
    --split val \
    --sample-count 300 \
    --rotations-per-sample 1 \
    --full-geometry-per-sample 1 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16 \
    --output "$D2R_CANDIDATE_REPORT"

if env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/evaluate_teacher_capacity_report.py \
    "$D2R_CANDIDATE_REPORT" --require-pass; then
    echo "D2R_V33_CANDIDATE_399999_CAPACITY_PASS report=$D2R_CANDIDATE_REPORT sha256=$candidate_sha"
else
    status=$?
    echo "D2R_V33_CANDIDATE_399999_CAPACITY_REJECT report=$D2R_CANDIDATE_REPORT sha256=$candidate_sha" >&2
    exit "$status"
fi
