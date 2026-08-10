#!/usr/bin/env bash
# Slurm visualization body: 1 GPU, 16 CPU, 30 minutes recommended.
# Replays one real v6 boundary sample and writes a two-row explanatory figure.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_C4_VIS_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_C4_VIS_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_C4_VIS_V6_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_v6/candidate-399999.json"
readonly D2R_C4_VIS_V6_SHA256="abf55cc22ea65665d175563c73d18ce993d7401923c4afe4e824073900655176"
readonly D2R_C4_VIS_SAMPLE_ID="Pers_NoAug_0010947"
readonly D2R_C4_VIS_RUN_TAG="job-${SLURM_JOB_ID:-manual}"
readonly D2R_C4_VIS_OUTPUT="runs/v33_diagnostics/teacher_c4_route_failure_visualization_v1/${D2R_C4_VIS_RUN_TAG}"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_c4_route_failure_visualization_v1

[[ -f "$D2R_C4_VIS_TEACHER" && -s "$D2R_C4_VIS_TEACHER" \
    && ! -L "$D2R_C4_VIS_TEACHER" ]] \
    || d2r_fail "399999 visualization teacher 不存在、为空或为符号链接：$D2R_C4_VIS_TEACHER"
teacher_digest_line="$(sha256sum -- "$D2R_C4_VIS_TEACHER")" \
    || d2r_fail "无法计算 visualization teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_C4_VIS_TEACHER_SHA256" ]] \
    || d2r_fail "visualization teacher SHA-256 已变化：expected=$D2R_C4_VIS_TEACHER_SHA256 actual=$teacher_sha"

[[ -f "$D2R_C4_VIS_V6_REPORT" && -s "$D2R_C4_VIS_V6_REPORT" \
    && ! -L "$D2R_C4_VIS_V6_REPORT" ]] \
    || d2r_fail "绑定的 best-of-C4 v6 报告不存在、为空或为符号链接：$D2R_C4_VIS_V6_REPORT"
v6_digest_line="$(sha256sum -- "$D2R_C4_VIS_V6_REPORT")" \
    || d2r_fail "无法计算 best-of-C4 v6 报告 SHA-256"
v6_sha="${v6_digest_line%% *}"
[[ "$v6_sha" == "$D2R_C4_VIS_V6_SHA256" ]] \
    || d2r_fail "best-of-C4 v6 报告 SHA-256 已变化：expected=$D2R_C4_VIS_V6_SHA256 actual=$v6_sha"
[[ ! -e "$D2R_C4_VIS_OUTPUT" && ! -L "$D2R_C4_VIS_OUTPUT" ]] \
    || d2r_fail "可视化输出目录已存在；请保留旧结果并使用新的 Slurm job id：$D2R_C4_VIS_OUTPUT"

mkdir -p "$(dirname -- "$D2R_C4_VIS_OUTPUT")"
echo "[info] 开始重放真实 C4 边界样本：sample=$D2R_C4_VIS_SAMPLE_ID output=$D2R_C4_VIS_OUTPUT"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/visualize_teacher_c4_route_failure.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_C4_VIS_TEACHER" \
    --expected-teacher-sha256 "$D2R_C4_VIS_TEACHER_SHA256" \
    --v6-report "$D2R_C4_VIS_V6_REPORT" \
    --expected-v6-report-sha256 "$D2R_C4_VIS_V6_SHA256" \
    --output-dir "$D2R_C4_VIS_OUTPUT" \
    --sample-id "$D2R_C4_VIS_SAMPLE_ID" \
    --device cuda:0 \
    --threads 16 \
    --residual-target-iterations 24 \
    --max-residual-px 40 \
    --feature-stride 8

readonly D2R_C4_VIS_REPORT="$D2R_C4_VIS_OUTPUT/report.json"
readonly D2R_C4_VIS_FIGURE="$D2R_C4_VIS_OUTPUT/c4_route_comparison.png"
[[ -f "$D2R_C4_VIS_REPORT" && -s "$D2R_C4_VIS_REPORT" \
    && ! -L "$D2R_C4_VIS_REPORT" ]] \
    || d2r_fail "可视化 CLI 未生成普通非空报告：$D2R_C4_VIS_REPORT"
[[ -f "$D2R_C4_VIS_FIGURE" && -s "$D2R_C4_VIS_FIGURE" \
    && ! -L "$D2R_C4_VIS_FIGURE" ]] \
    || d2r_fail "可视化 CLI 未生成普通非空总图：$D2R_C4_VIS_FIGURE"

report_identity="$(
    "$D2R_PYTHON" -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); p=d.get("protocol",{}); s=d.get("sample",{}); r=d.get("routes",{}); w=r.get("wrong",{}); c=r.get("correct",{}); print(d.get("kind"), d.get("report_version"), d.get("diagnostic_only"), d.get("can_approve_production"), d.get("actual_teacher_outputs"), d.get("uses_ground_truth_flow_for_oracle_projection"), s.get("id"), w.get("quarter_turn_deg"), c.get("quarter_turn_deg"), p.get("residual_target_iterations"), p.get("max_residual_px"), p.get("feature_stride"), len(d.get("artifacts",[])), sep="\t")' \
        "$D2R_C4_VIS_REPORT"
)" || d2r_fail "无法复验 C4 route visualization 报告"
IFS=$'\t' read -r report_kind report_version diagnostic_only can_approve actual_teacher uses_gt sample_id wrong_turn correct_turn iterations cap stride artifact_count extra \
    <<<"$report_identity"
[[ "$report_kind" == "teacher_c4_route_failure_visualization" \
    && "$report_version" == "1" \
    && "$diagnostic_only" == "True" \
    && "$can_approve" == "False" \
    && "$actual_teacher" == "True" \
    && "$uses_gt" == "True" \
    && "$sample_id" == "$D2R_C4_VIS_SAMPLE_ID" \
    && "$wrong_turn" == "0" \
    && "$correct_turn" == "90" \
    && "$iterations" == "24" \
    && "$cap" == "40.0" \
    && "$stride" == "8" \
    && "$artifact_count" == "9" \
    && -z "${extra:-}" ]] \
    || d2r_fail "C4 route visualization 报告身份、协议或产物数不符合预期：$report_identity"

echo "D2R_V33_C4_ROUTE_VISUALIZATION_COMPLETE figure=$D2R_C4_VIS_FIGURE report=$D2R_C4_VIS_REPORT teacher_sha256=$teacher_sha v6_sha256=$v6_sha"
