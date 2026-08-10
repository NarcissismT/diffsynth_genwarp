#!/usr/bin/env bash
# Portal resources: CPU job, 2 CPU, 8G RAM. Run only after metric/visual review.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

: "${DOCGRID_GATE:?export DOCGRID_GATE=gate1..gate5}"
export DOCGRID_SEED="${DOCGRID_SEED:-1337}"
case "${DOCGRID_GATE}" in
  gate1)
    checkpoint_stage=stage1_coarse
    default_baseline="${DOCGRID_RUN_ROOT}/baselines/frozen_supervised_prior/metrics.json"
    ;;
  gate2)
    checkpoint_stage=stage2_warr
    default_baseline="${DOCGRID_RUN_ROOT}/stage1_coarse/seed-${DOCGRID_SEED}/gate1_eval/metrics.json"
    ;;
  gate3)
    checkpoint_stage=stage3_coordinate_fm
    default_baseline="${DOCGRID_RUN_ROOT}/stage2_warr/seed-${DOCGRID_SEED}/gate2_eval/metrics.json"
    ;;
  gate4)
    checkpoint_stage=stage4_qwen
    default_baseline="${DOCGRID_RUN_ROOT}/stage3_coordinate_fm/seed-${DOCGRID_SEED}/gate3_eval/metrics.json"
    ;;
  gate5)
    checkpoint_stage=stage5_full_page
    default_baseline="${DOCGRID_RUN_ROOT}/baselines/full_page_deterministic/seed-${DOCGRID_SEED}/metrics.json"
    ;;
  *) echo "DOCGRID_GATE must be gate1..gate5" >&2; exit 2 ;;
esac
export DOCGRID_EVAL_OUTPUT="${DOCGRID_EVAL_OUTPUT:-${DOCGRID_RUN_ROOT}/${checkpoint_stage}/seed-${DOCGRID_SEED}/${DOCGRID_GATE}_eval}"
export DOCGRID_GATE_RECEIPT="${DOCGRID_GATE_RECEIPT:-${DOCGRID_RUN_ROOT}/gates/${DOCGRID_GATE}.json}"
export DOCGRID_BASELINE_EVALUATION="${DOCGRID_BASELINE_EVALUATION:-${default_baseline}}"
: "${DOCGRID_GATE_REVIEWER:?export DOCGRID_GATE_REVIEWER}"
: "${DOCGRID_GATE_REVIEW_NOTE:?export DOCGRID_GATE_REVIEW_NOTE}"
: "${DOCGRID_GATE_DECISION:?export DOCGRID_GATE_DECISION=passed|failed}"

docgrid_run_container 2 \
  'bash cp_docflow_v1/slurm/docgrid_v2/container_jobs/write_gate_receipt_worker.sh'
