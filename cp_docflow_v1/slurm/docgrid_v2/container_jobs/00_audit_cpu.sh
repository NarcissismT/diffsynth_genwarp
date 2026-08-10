#!/usr/bin/env bash
# Portal resources: CPU job, 16 CPU, 64G host RAM.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

readonly merged_root="${DOCGRID_ANALYTIC_MERGED_OUTPUT:-${DOCGRID_DATA_ROOT}/analytic_merged_seed1337_512}"
export DOCGRID_TRAIN_MANIFEST="${DOCGRID_TRAIN_MANIFEST:-${merged_root}/manifests/train.jsonl}"
export DOCGRID_VAL_MANIFEST="${DOCGRID_VAL_MANIFEST:-${merged_root}/manifests/val.jsonl}"
export DOCGRID_TEST_MANIFEST="${DOCGRID_TEST_MANIFEST:-${merged_root}/manifests/test.jsonl}"
export DOCGRID_AUDIT_OUTPUT="${DOCGRID_AUDIT_OUTPUT:-${DOCGRID_RUN_ROOT}/stage0_audit}"
export DOCGRID_FROZEN_CONTRACT="${DOCGRID_FROZEN_CONTRACT:-${DOCGRID_AUDIT_OUTPUT}/frozen_contract.json}"
export DOCGRID_BASELINE_CHECKPOINT="${DOCGRID_BASELINE_CHECKPOINT:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/259999_raft_unwarp.pt}"
export DOCGRID_BASELINE_CONFIG="${DOCGRID_BASELINE_CONFIG:-${DOCGRID_CONTAINER_PROJECT_ROOT}/configs/docgrid_v2/frozen_supervised_prior_baseline.yaml}"
export DOCGRID_BASELINE_METRICS="${DOCGRID_BASELINE_METRICS:-${DOCGRID_RUN_ROOT}/baselines/frozen_supervised_prior/metrics.json}"

docgrid_run_container 16 \
  'bash cp_docflow_v1/slurm/docgrid_v2/audit_stage0.sbatch'
