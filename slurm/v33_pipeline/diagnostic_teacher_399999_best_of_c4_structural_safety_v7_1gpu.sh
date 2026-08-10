#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 2 hours recommended.
# Replays the v6 GT-selected C4 label once per sample and audits the 24x40
# stride-8 upper bound for in-bounds, topology, line, curvature, and texture.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_STRUCTURAL_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_STRUCTURAL_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_BOUND_BEST_C4_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_v6/candidate-399999.json"
readonly D2R_BOUND_BEST_C4_REPORT_SHA256="abf55cc22ea65665d175563c73d18ce993d7401923c4afe4e824073900655176"
readonly D2R_STRUCTURAL_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_structural_safety_v7/candidate-399999.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_best_of_c4_structural_safety_v7_diagnostic

for input_path in "$D2R_STRUCTURAL_TEACHER" "$D2R_BOUND_BEST_C4_REPORT"; do
    [[ -f "$input_path" && -s "$input_path" && ! -L "$input_path" ]] \
        || d2r_fail "v7 输入不存在、为空或为符号链接：$input_path"
done
teacher_digest_line="$(sha256sum -- "$D2R_STRUCTURAL_TEACHER")" \
    || d2r_fail "无法计算 399999 structural teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_STRUCTURAL_TEACHER_SHA256" ]] \
    || d2r_fail "399999 structural teacher SHA-256 已变化：expected=$D2R_STRUCTURAL_TEACHER_SHA256 actual=$teacher_sha"
v6_digest_line="$(sha256sum -- "$D2R_BOUND_BEST_C4_REPORT")" \
    || d2r_fail "无法计算绑定的 best-of-C4 v6 报告 SHA-256"
v6_sha="${v6_digest_line%% *}"
[[ "$v6_sha" == "$D2R_BOUND_BEST_C4_REPORT_SHA256" ]] \
    || d2r_fail "best-of-C4 v6 报告已变化：expected=$D2R_BOUND_BEST_C4_REPORT_SHA256 actual=$v6_sha"
[[ ! -e "$D2R_STRUCTURAL_REPORT" && ! -L "$D2R_STRUCTURAL_REPORT" ]] \
    || d2r_fail "structural safety v7 报告已存在；请先改名保存后再重跑：$D2R_STRUCTURAL_REPORT"

mkdir -p "$(dirname -- "$D2R_STRUCTURAL_REPORT")"
echo "[info] 开始 diagnostic-only structural safety：v6 best-EPE C4，24 iterations，24/32/40 px"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/diagnose_teacher_best_of_c4_structural_safety.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_STRUCTURAL_TEACHER" \
    --expected-teacher-sha256 "$D2R_STRUCTURAL_TEACHER_SHA256" \
    --best-of-c4-report "$D2R_BOUND_BEST_C4_REPORT" \
    --output "$D2R_STRUCTURAL_REPORT" \
    --sample-count 300 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16 \
    --residual-target-iterations 24 \
    --residual-cap-sweep 24 32 40 \
    --baseline-max-residual-px 24 \
    --selected-max-residual-px 40

[[ -f "$D2R_STRUCTURAL_REPORT" && -s "$D2R_STRUCTURAL_REPORT" \
    && ! -L "$D2R_STRUCTURAL_REPORT" ]] \
    || d2r_fail "structural safety v7 CLI 未生成普通非空报告：$D2R_STRUCTURAL_REPORT"
report_identity="$("$D2R_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); i=d.get("identities",{}); p=d.get("protocol",{}); r=d.get("results",{}); s=r.get("selected_cell",{}); g=r.get("structural_grid",[]); x=r.get("decision",{}); print(d.get("kind"), d.get("report_version"), d.get("diagnostic_only"), d.get("can_approve_production"), d.get("uses_ground_truth_flow_for_candidate_selection"), i.get("bound_best_of_c4_v6_report",{}).get("sha256"), p.get("sample_count"), p.get("residual_target_iterations"), ",".join(str(v) for v in p.get("residual_cap_sweep_px",[])), p.get("baseline_max_residual_px"), p.get("selected_max_residual_px"), s.get("residual_target_iterations"), s.get("max_residual_px"), len(g), x.get("policy_id"), x.get("passed"), x.get("can_approve_production"), len(r.get("samples",[])), sep="\t")' \
    "$D2R_STRUCTURAL_REPORT")" || d2r_fail "无法复验 structural safety v7 报告"
IFS=$'\t' read -r report_kind report_version diagnostic_only can_approve uses_gt_flow bound_v6_sha report_samples protocol_iterations protocol_caps baseline_cap selected_cap cell_iterations cell_cap grid_cells policy_id structural_pass decision_can_approve result_samples extra \
    <<<"$report_identity"
[[ "$report_kind" == "teacher_quarter_turn_oracle_canonical_full_geometry_structural_safety_diagnostic" \
    && "$report_version" == "7" \
    && "$diagnostic_only" == "True" \
    && "$can_approve" == "False" \
    && "$uses_gt_flow" == "True" \
    && "$bound_v6_sha" == "$D2R_BOUND_BEST_C4_REPORT_SHA256" \
    && "$report_samples" == "300" \
    && "$protocol_iterations" == "24" \
    && "$protocol_caps" == "24.0,32.0,40.0" \
    && "$baseline_cap" == "24.0" \
    && "$selected_cap" == "40.0" \
    && "$cell_iterations" == "24" \
    && "$cell_cap" == "40.0" \
    && "$grid_cells" == "3" \
    && "$policy_id" == "best_of_c4_stride8_structural_safety_v1" \
    && "$decision_can_approve" == "False" \
    && "$result_samples" == "300" \
    && -z "${extra:-}" ]] \
    || d2r_fail "structural safety v7 报告身份、协议或样本数不符合预期：$report_identity"

echo "D2R_V33_BEST_OF_C4_STRUCTURAL_SAFETY_COMPLETE report=$D2R_STRUCTURAL_REPORT passed=$structural_pass teacher_sha256=$teacher_sha bound_v6_sha256=$v6_sha"
