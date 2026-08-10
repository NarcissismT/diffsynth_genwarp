#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 1 hour recommended.
# Uses injected rotation metadata and cannot approve or launch production.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_CANONICAL_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_CANONICAL_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_CANONICAL_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_frame_v2/candidate-399999.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_oracle_c4_canonical_frame_v2_diagnostic

[[ -f "$D2R_CANONICAL_TEACHER" && -s "$D2R_CANONICAL_TEACHER" \
    && ! -L "$D2R_CANONICAL_TEACHER" ]] \
    || d2r_fail "399999 canonical-frame teacher 不存在、为空或为符号链接：$D2R_CANONICAL_TEACHER"
teacher_digest_line="$(sha256sum -- "$D2R_CANONICAL_TEACHER")" \
    || d2r_fail "无法计算 399999 canonical-frame teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_CANONICAL_TEACHER_SHA256" ]] \
    || d2r_fail "399999 canonical-frame teacher SHA-256 已变化：expected=$D2R_CANONICAL_TEACHER_SHA256 actual=$teacher_sha"
[[ ! -e "$D2R_CANONICAL_REPORT" && ! -L "$D2R_CANONICAL_REPORT" ]] \
    || d2r_fail "canonical-frame v2 诊断报告已存在；请先改名保存后再重跑：$D2R_CANONICAL_REPORT"

mkdir -p "$(dirname -- "$D2R_CANONICAL_REPORT")"
echo "[info] 开始 diagnostic-only canonical-frame v2：C4 normalize -> teacher/residual audit -> final map-back control"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/diagnose_teacher_quarter_turn_oracle_canonical_frame.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_CANONICAL_TEACHER" \
    --expected-teacher-sha256 "$D2R_CANONICAL_TEACHER_SHA256" \
    --output "$D2R_CANONICAL_REPORT" \
    --sample-count 300 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16

[[ -f "$D2R_CANONICAL_REPORT" && -s "$D2R_CANONICAL_REPORT" \
    && ! -L "$D2R_CANONICAL_REPORT" ]] \
    || d2r_fail "canonical-frame v2 CLI 未生成普通非空报告：$D2R_CANONICAL_REPORT"
report_identity="$(
    "$D2R_PYTHON" -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d.get("kind"), d.get("report_version"), d.get("diagnostic_only"), d.get("can_approve_production"), d.get("results",{}).get("canonical_rotation_augmented",{}).get("sample_count"), sep="\t")' \
        "$D2R_CANONICAL_REPORT"
)" || d2r_fail "无法复验 canonical-frame v2 报告"
IFS=$'\t' read -r report_kind report_version diagnostic_only can_approve sample_count extra \
    <<<"$report_identity"
[[ "$report_kind" == "teacher_quarter_turn_oracle_canonical_frame_diagnostic" \
    && "$report_version" == "2" \
    && "$diagnostic_only" == "True" \
    && "$can_approve" == "False" \
    && "$sample_count" == "300" \
    && -z "${extra:-}" ]] \
    || d2r_fail "canonical-frame v2 报告身份或样本数不符合预期：$report_identity"

echo "D2R_V33_QUARTER_TURN_CANONICAL_FRAME_DIAGNOSTIC_COMPLETE report=$D2R_CANONICAL_REPORT teacher_sha256=$teacher_sha"
