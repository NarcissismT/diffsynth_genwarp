#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU; use 192G host RAM for Gate 4/5, 64G otherwise.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

: "${DOCGRID_GATE:?export DOCGRID_GATE=gate1..gate5 before submission}"
export DOCGRID_SEED="${DOCGRID_SEED:-1337}"
readonly merged_root="${DOCGRID_ANALYTIC_MERGED_OUTPUT:-${DOCGRID_DATA_ROOT}/analytic_merged_seed1337_512}"

case "${DOCGRID_GATE}" in
  gate1) checkpoint_stage=stage1_coarse ;;
  gate2) checkpoint_stage=stage2_warr ;;
  gate3) checkpoint_stage=stage3_coordinate_fm ;;
  gate4) checkpoint_stage=stage4_qwen ;;
  gate5) checkpoint_stage=stage5_full_page ;;
  *) echo "DOCGRID_GATE must be gate1..gate5" >&2; exit 2 ;;
esac
if [[ "${DOCGRID_GATE}" == "gate5" ]]; then
  export DOCGRID_EVAL_MANIFEST="${DOCGRID_EVAL_MANIFEST:-${DOCGRID_STAGE5_VAL_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/val.jsonl}}"
else
  export DOCGRID_EVAL_MANIFEST="${DOCGRID_EVAL_MANIFEST:-${merged_root}/manifests/val.jsonl}"
fi
export DOCGRID_CHECKPOINT="${DOCGRID_CHECKPOINT:-${DOCGRID_RUN_ROOT}/${checkpoint_stage}/seed-${DOCGRID_SEED}/best.pt}"
export DOCGRID_EVAL_OUTPUT="${DOCGRID_EVAL_OUTPUT:-${DOCGRID_RUN_ROOT}/${checkpoint_stage}/seed-${DOCGRID_SEED}/${DOCGRID_GATE}_eval}"

# This produces immutable metrics/visuals only. Passing receipts remain a
# separate reviewed action and are never auto-created by this script.
docgrid_run_container 8 \
  'bash cp_docflow_v1/slurm/docgrid_v2/evaluate_gate.sbatch'
