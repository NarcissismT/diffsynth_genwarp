#!/usr/bin/env bash
# Portal resources: CPU job, 16 CPU, 64G host RAM. Requires full-page manifests.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

export DOCGRID_TRAIN_MANIFEST="${DOCGRID_STAGE5_TRAIN_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/train.jsonl}"
export DOCGRID_VAL_MANIFEST="${DOCGRID_STAGE5_VAL_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/val.jsonl}"
export DOCGRID_TEST_MANIFEST="${DOCGRID_STAGE5_TEST_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/test.jsonl}"
export DOCGRID_AUDIT_OUTPUT="${DOCGRID_AUDIT_OUTPUT:-${DOCGRID_RUN_ROOT}/stage0_full_page_audit}"
export DOCGRID_FROZEN_CONTRACT="${DOCGRID_FROZEN_CONTRACT:-${DOCGRID_AUDIT_OUTPUT}/frozen_contract.json}"

docgrid_run_container 16 \
  'bash cp_docflow_v1/slurm/docgrid_v2/audit_stage0.sbatch'
