#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 1 hour recommended.
# Uses injected rotation metadata as an oracle; it cannot approve production.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_ORACLE_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_ORACLE_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_ORACLE_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle/candidate-399999.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_oracle_c4_diagnostic

[[ -f "$D2R_ORACLE_TEACHER" && -s "$D2R_ORACLE_TEACHER" \
    && ! -L "$D2R_ORACLE_TEACHER" ]] \
    || d2r_fail "399999 oracle teacher 不存在、为空或为符号链接：$D2R_ORACLE_TEACHER"
teacher_digest_line="$(sha256sum -- "$D2R_ORACLE_TEACHER")" \
    || d2r_fail "无法计算 399999 oracle teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_ORACLE_TEACHER_SHA256" ]] \
    || d2r_fail "399999 oracle teacher SHA-256 已变化：expected=$D2R_ORACLE_TEACHER_SHA256 actual=$teacher_sha"
[[ ! -e "$D2R_ORACLE_REPORT" ]] \
    || d2r_fail "oracle C4 诊断报告已存在；请先改名保存后再重跑：$D2R_ORACLE_REPORT"

mkdir -p "$(dirname -- "$D2R_ORACLE_REPORT")"
echo "[info] 开始 diagnostic-only oracle C4：known rotation -> quarter-turn normalization -> teacher -> map-back"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/diagnose_teacher_quarter_turn_oracle.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_ORACLE_TEACHER" \
    --expected-teacher-sha256 "$D2R_ORACLE_TEACHER_SHA256" \
    --output "$D2R_ORACLE_REPORT" \
    --sample-count 300 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16

echo "D2R_V33_QUARTER_TURN_DIAGNOSTIC_COMPLETE report=$D2R_ORACLE_REPORT teacher_sha256=$teacher_sha"
