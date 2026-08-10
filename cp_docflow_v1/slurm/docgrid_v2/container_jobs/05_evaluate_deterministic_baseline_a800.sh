#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU, 96G host RAM, 24 hours.
# Run after Stage 2 and full-page Stage-0 audit; required only for Gate 5.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

export DOCGRID_SEED="${DOCGRID_SEED:-1337}"
export DOCGRID_CHECKPOINT="${DOCGRID_CHECKPOINT:-${DOCGRID_RUN_ROOT}/stage2_warr/seed-${DOCGRID_SEED}/best.pt}"
export DOCGRID_EVAL_MANIFEST="${DOCGRID_EVAL_MANIFEST:-${DOCGRID_STAGE5_VAL_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/val.jsonl}}"
export DOCGRID_STAGE5_FROZEN_CONTRACT="${DOCGRID_STAGE5_FROZEN_CONTRACT:-${DOCGRID_RUN_ROOT}/stage0_full_page_audit/frozen_contract.json}"
export DOCGRID_BASELINE_EVALUATION_OUTPUT="${DOCGRID_BASELINE_EVALUATION_OUTPUT:-${DOCGRID_RUN_ROOT}/baselines/full_page_deterministic/seed-${DOCGRID_SEED}}"
export DOCGRID_EVAL_INPUT_HEIGHT="${DOCGRID_EVAL_INPUT_HEIGHT:-1024}"
export DOCGRID_EVAL_INPUT_WIDTH="${DOCGRID_EVAL_INPUT_WIDTH:-768}"
export DOCGRID_EVAL_OUTPUT_HEIGHT="${DOCGRID_EVAL_OUTPUT_HEIGHT:-1024}"
export DOCGRID_EVAL_OUTPUT_WIDTH="${DOCGRID_EVAL_OUTPUT_WIDTH:-768}"

docgrid_run_container 8 \
  'bash cp_docflow_v1/slurm/docgrid_v2/evaluate_deterministic_full_page_baseline.sbatch'
