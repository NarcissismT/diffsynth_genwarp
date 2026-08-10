#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU, 64G host RAM, 24 hours.
# Run after 00_merge_cpu.sh and before 00_audit_cpu.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

readonly merged_root="${DOCGRID_ANALYTIC_MERGED_OUTPUT:-${DOCGRID_DATA_ROOT}/analytic_merged_seed1337_512}"
export DOCGRID_EVAL_MANIFEST="${DOCGRID_EVAL_MANIFEST:-${merged_root}/manifests/val.jsonl}"
export DOCGRID_BASELINE_CONFIG="${DOCGRID_BASELINE_CONFIG:-${DOCGRID_CONTAINER_PROJECT_ROOT}/configs/docgrid_v2/frozen_supervised_prior_baseline.yaml}"
export DOCGRID_BASELINE_CHECKPOINT="${DOCGRID_BASELINE_CHECKPOINT:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/259999_raft_unwarp.pt}"
export DOCGRID_BASELINE_EVALUATION_OUTPUT="${DOCGRID_BASELINE_EVALUATION_OUTPUT:-${DOCGRID_RUN_ROOT}/baselines/frozen_supervised_prior}"
export DOCGRID_BASELINE_METRICS="${DOCGRID_BASELINE_METRICS:-${DOCGRID_BASELINE_EVALUATION_OUTPUT}/metrics.json}"

docgrid_run_container 8 \
  'bash cp_docflow_v1/slurm/docgrid_v2/evaluate_frozen_prior_baseline.sbatch'
