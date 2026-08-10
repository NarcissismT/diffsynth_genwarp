#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 1 hour recommended.
# Uses injected full-geometry metadata and cannot approve or launch production.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_FULL_GEOMETRY_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_FULL_GEOMETRY_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_FULL_GEOMETRY_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_v4/candidate-399999.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_oracle_c4_canonical_full_geometry_v4_diagnostic

[[ -f "$D2R_FULL_GEOMETRY_TEACHER" && -s "$D2R_FULL_GEOMETRY_TEACHER" \
    && ! -L "$D2R_FULL_GEOMETRY_TEACHER" ]] \
    || d2r_fail "399999 canonical full-geometry teacher 不存在、为空或为符号链接：$D2R_FULL_GEOMETRY_TEACHER"
teacher_digest_line="$(sha256sum -- "$D2R_FULL_GEOMETRY_TEACHER")" \
    || d2r_fail "无法计算 399999 canonical full-geometry teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_FULL_GEOMETRY_TEACHER_SHA256" ]] \
    || d2r_fail "399999 canonical full-geometry teacher SHA-256 已变化：expected=$D2R_FULL_GEOMETRY_TEACHER_SHA256 actual=$teacher_sha"
[[ ! -e "$D2R_FULL_GEOMETRY_REPORT" && ! -L "$D2R_FULL_GEOMETRY_REPORT" ]] \
    || d2r_fail "canonical full-geometry v4 报告已存在；请先改名保存后再重跑：$D2R_FULL_GEOMETRY_REPORT"

mkdir -p "$(dirname -- "$D2R_FULL_GEOMETRY_REPORT")"
echo "[info] 开始 diagnostic-only full geometry：formal H -> oracle C4 -> teacher -> 12-step residual audit"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/diagnose_teacher_full_geometry_oracle_canonical.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_FULL_GEOMETRY_TEACHER" \
    --expected-teacher-sha256 "$D2R_FULL_GEOMETRY_TEACHER_SHA256" \
    --output "$D2R_FULL_GEOMETRY_REPORT" \
    --sample-count 300 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16

[[ -f "$D2R_FULL_GEOMETRY_REPORT" && -s "$D2R_FULL_GEOMETRY_REPORT" \
    && ! -L "$D2R_FULL_GEOMETRY_REPORT" ]] \
    || d2r_fail "canonical full-geometry v4 CLI 未生成普通非空报告：$D2R_FULL_GEOMETRY_REPORT"
report_identity="$(
    "$D2R_PYTHON" -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); p=d.get("protocol",{}); f=p.get("source_full_geometry",{}); r=d.get("results",{}); s=r.get("samples",[]); print(d.get("kind"), d.get("report_version"), d.get("diagnostic_only"), d.get("can_approve_production"), p.get("configured_residual_target_iterations"), p.get("residual_target_iterations"), f.get("transformations_per_sample"), f.get("seed"), r.get("canonical_full_geometry_augmented",{}).get("sample_count"), len(s), all(x.get("full_geometry_seed") is not None and x.get("source_homography") is not None for x in s), sep="\t")' \
        "$D2R_FULL_GEOMETRY_REPORT"
)" || d2r_fail "无法复验 canonical full-geometry v4 报告"
IFS=$'\t' read -r report_kind report_version diagnostic_only can_approve configured_iterations effective_iterations transformations_per_sample protocol_seed aggregate_count sample_count metadata_complete extra \
    <<<"$report_identity"
[[ "$report_kind" == "teacher_quarter_turn_oracle_canonical_full_geometry_diagnostic" \
    && "$report_version" == "4" \
    && "$diagnostic_only" == "True" \
    && "$can_approve" == "False" \
    && "$configured_iterations" == "6" \
    && "$effective_iterations" == "12" \
    && "$transformations_per_sample" == "1" \
    && "$protocol_seed" == "42" \
    && "$aggregate_count" == "300" \
    && "$sample_count" == "300" \
    && "$metadata_complete" == "True" \
    && -z "${extra:-}" ]] \
    || d2r_fail "canonical full-geometry v4 报告身份、协议或样本数不符合预期：$report_identity"

echo "D2R_V33_QUARTER_TURN_CANONICAL_FULL_GEOMETRY_COMPLETE report=$D2R_FULL_GEOMETRY_REPORT teacher_sha256=$teacher_sha"
