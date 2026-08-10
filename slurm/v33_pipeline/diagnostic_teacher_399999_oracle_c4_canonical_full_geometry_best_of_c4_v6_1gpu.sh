#!/usr/bin/env bash
# Slurm diagnostic body: 1 GPU, 16 CPU, 3 hours recommended.
# Runs four real teacher forwards per frozen full-geometry sample, then uses GT
# only to compare nearest-angle and best-of-C4 upper bounds.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly D2R_BEST_C4_TEACHER="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt"
readonly D2R_BEST_C4_TEACHER_SHA256="e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5"
readonly D2R_BEST_C4_REPORT="runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_v6/candidate-399999.json"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock candidate_399999_oracle_c4_canonical_full_geometry_best_of_c4_v6_diagnostic

[[ -f "$D2R_BEST_C4_TEACHER" && -s "$D2R_BEST_C4_TEACHER" \
    && ! -L "$D2R_BEST_C4_TEACHER" ]] \
    || d2r_fail "399999 best-of-C4 teacher 不存在、为空或为符号链接：$D2R_BEST_C4_TEACHER"
teacher_digest_line="$(sha256sum -- "$D2R_BEST_C4_TEACHER")" \
    || d2r_fail "无法计算 399999 best-of-C4 teacher SHA-256"
teacher_sha="${teacher_digest_line%% *}"
[[ "$teacher_sha" == "$D2R_BEST_C4_TEACHER_SHA256" ]] \
    || d2r_fail "399999 best-of-C4 teacher SHA-256 已变化：expected=$D2R_BEST_C4_TEACHER_SHA256 actual=$teacher_sha"
[[ ! -e "$D2R_BEST_C4_REPORT" && ! -L "$D2R_BEST_C4_REPORT" ]] \
    || d2r_fail "best-of-C4 v6 报告已存在；请先改名保存后再重跑：$D2R_BEST_C4_REPORT"

mkdir -p "$(dirname -- "$D2R_BEST_C4_REPORT")"
echo "[info] 开始 diagnostic-only best-of-C4：300 full-geometry x 4 real teacher candidates"
env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
    "$D2R_PYTHON" scripts/diagnose_teacher_full_geometry_oracle_best_of_c4.py \
    --config "$D2R_CONFIG" \
    --checkpoint "$D2R_SEED_CHECKPOINT" \
    --teacher "$D2R_BEST_C4_TEACHER" \
    --expected-teacher-sha256 "$D2R_BEST_C4_TEACHER_SHA256" \
    --output "$D2R_BEST_C4_REPORT" \
    --sample-count 300 \
    --seed 42 \
    --batch-size 1 \
    --device cuda:0 \
    --threads 16

[[ -f "$D2R_BEST_C4_REPORT" && -s "$D2R_BEST_C4_REPORT" \
    && ! -L "$D2R_BEST_C4_REPORT" ]] \
    || d2r_fail "best-of-C4 v6 CLI 未生成普通非空报告：$D2R_BEST_C4_REPORT"
report_identity="$(
    "$D2R_PYTHON" -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); p=d.get("protocol",{}); r=d.get("results",{}); b=p.get("best_of_c4",{}); g=r.get("capacity_grid",[]); e=r.get("best_teacher_epe_capacity_grid",[]); a=r.get("best_capacity_aware_capacity_grid",[]); c=r.get("all_candidate_capacity_grid",[]); s=r.get("samples",[]); rc=r.get("routing_comparison",{}); print(d.get("kind"), d.get("report_version"), d.get("diagnostic_only"), d.get("can_approve_production"), d.get("uses_ground_truth_flow_for_candidate_selection"), b.get("candidate_batch_size"), ",".join(str(x) for x in b.get("candidate_order",[])), len(g), len(e), len(a), len(c), len(s), rc.get("sample_count"), min((len(x.get("c4_best_of_four",{}).get("candidates",[])) for x in s), default=-1), max((len(x.get("c4_best_of_four",{}).get("candidates",[])) for x in s), default=-1), sep="\t")' \
        "$D2R_BEST_C4_REPORT"
)" || d2r_fail "无法复验 best-of-C4 v6 报告"
IFS=$'\t' read -r report_kind report_version diagnostic_only can_approve uses_gt_flow candidate_batch candidate_order nearest_cells epe_cells capacity_cells all_cells report_samples routing_samples min_candidates max_candidates extra \
    <<<"$report_identity"
[[ "$report_kind" == "teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_diagnostic" \
    && "$report_version" == "6" \
    && "$diagnostic_only" == "True" \
    && "$can_approve" == "False" \
    && "$uses_gt_flow" == "True" \
    && "$candidate_batch" == "1" \
    && "$candidate_order" == "0,-90,90,180" \
    && "$nearest_cells" == "6" \
    && "$epe_cells" == "6" \
    && "$capacity_cells" == "6" \
    && "$all_cells" == "6" \
    && "$report_samples" == "300" \
    && "$routing_samples" == "300" \
    && "$min_candidates" == "4" \
    && "$max_candidates" == "4" \
    && -z "${extra:-}" ]] \
    || d2r_fail "best-of-C4 v6 报告身份、协议或样本数不符合预期：$report_identity"

echo "D2R_V33_QUARTER_TURN_CANONICAL_FULL_GEOMETRY_BEST_OF_C4_COMPLETE report=$D2R_BEST_C4_REPORT teacher_sha256=$teacher_sha"
