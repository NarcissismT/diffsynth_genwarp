#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 1 hour recommended.
# Reuses each oracle-canonical teacher forward for a 2x3 offline capacity grid.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_CAPACITY_GRID_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_CAPACITY_GRID_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_CAPACITY_GRID_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_v5/candidate-399999.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_oracle_c4_canonical_full_geometry_capacity_grid_v5_diagnostic

[[ -f "$D2R_CAPACITY_GRID_TEACHER" && -s "$D2R_CAPACITY_GRID_TEACHER" \
    && ! -L "$D2R_CAPACITY_GRID_TEACHER" ]] \
    || d2r_fail "399999 full-geometry capacity-grid teacher 不存在、为空或为符号链接：$D2R_CAPACITY_GRID_TEACHER"
teacher_digest_line="$(sha256sum -- "$D2R_CAPACITY_GRID_TEACHER")" \
    || d2r_fail "无法计算 399999 full-geometry capacity-grid teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_CAPACITY_GRID_TEACHER_SHA256" ]] \
    || d2r_fail "399999 full-geometry capacity-grid teacher SHA-256 已变化：expected=$D2R_CAPACITY_GRID_TEACHER_SHA256 actual=$teacher_sha"
[[ ! -e "$D2R_CAPACITY_GRID_REPORT" && ! -L "$D2R_CAPACITY_GRID_REPORT" ]] \
    || d2r_fail "full-geometry capacity-grid v5 报告已存在；请先改名保存后再重跑：$D2R_CAPACITY_GRID_REPORT"

mkdir -p "$(dirname -- "$D2R_CAPACITY_GRID_REPORT")"
echo "[info] 开始 diagnostic-only capacity grid：solver=12/24 x residual_cap=24/32/40 px"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/diagnose_teacher_full_geometry_oracle_capacity_grid.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_CAPACITY_GRID_TEACHER" \
    --expected-teacher-sha256 "$D2R_CAPACITY_GRID_TEACHER_SHA256" \
    --output "$D2R_CAPACITY_GRID_REPORT" \
    --sample-count 300 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16

[[ -f "$D2R_CAPACITY_GRID_REPORT" && -s "$D2R_CAPACITY_GRID_REPORT" \
    && ! -L "$D2R_CAPACITY_GRID_REPORT" ]] \
    || d2r_fail "full-geometry capacity-grid v5 CLI 未生成普通非空报告：$D2R_CAPACITY_GRID_REPORT"
report_identity="$(
    "$D2R_PYTHON" -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); p=d.get("protocol",{}); r=d.get("results",{}); g=r.get("capacity_grid",[]); s=r.get("samples",[]); print(d.get("kind"), d.get("report_version"), d.get("diagnostic_only"), d.get("can_approve_production"), r.get("baseline_residual_target_iterations"), r.get("baseline_max_residual_px"), ",".join(str(x) for x in p.get("full_geometry_solver_iteration_sweep",[])), ",".join(f"{x:g}" for x in p.get("full_geometry_residual_cap_sweep_px",[])), ",".join("{}:{:g}".format(x.get("residual_target_iterations"), x.get("max_residual_px")) for x in g), ",".join(str(x.get("canonical_full_geometry_augmented",{}).get("sample_count")) for x in g), len(s), sep="\t")' \
        "$D2R_CAPACITY_GRID_REPORT"
)" || d2r_fail "无法复验 full-geometry capacity-grid v5 报告"
IFS=$'\t' read -r report_kind report_version diagnostic_only can_approve baseline_iterations baseline_cap protocol_iterations protocol_caps grid_cells sample_counts report_samples extra \
    <<<"$report_identity"
[[ "$report_kind" == "teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_diagnostic" \
    && "$report_version" == "5" \
    && "$diagnostic_only" == "True" \
    && "$can_approve" == "False" \
    && "$baseline_iterations" == "12" \
    && "$baseline_cap" == "24.0" \
    && "$protocol_iterations" == "12,24" \
    && "$protocol_caps" == "24,32,40" \
    && "$grid_cells" == "12:24,12:32,12:40,24:24,24:32,24:40" \
    && "$sample_counts" == "300,300,300,300,300,300" \
    && "$report_samples" == "300" \
    && -z "${extra:-}" ]] \
    || d2r_fail "full-geometry capacity-grid v5 报告身份、协议或样本数不符合预期：$report_identity"

echo "D2R_V33_QUARTER_TURN_CANONICAL_FULL_GEOMETRY_CAPACITY_GRID_COMPLETE report=$D2R_CAPACITY_GRID_REPORT teacher_sha256=$teacher_sha"
