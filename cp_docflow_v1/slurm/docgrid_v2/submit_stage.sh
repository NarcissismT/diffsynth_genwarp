#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stage="${1:?usage: submit_stage.sh STAGE [extra sbatch arguments]}"
shift
docgrid_preflight "${stage}" "${DOCGRID_SEED:-1337}"
mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"

resource_args=(--mem="${DOCGRID_STANDARD_HOST_MEM:-64G}")
case "${stage}" in
  stage4_qwen|stage5_full_page)
    # Frozen 20B Qwen with CPU offload needs substantially more host memory
    # than the deterministic stages. A caller's later --mem still wins.
    resource_args=(--mem="${DOCGRID_QWEN_HOST_MEM:-192G}")
    ;;
esac

sbatch \
  --job-name="docgrid-${stage}" \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/${stage}-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/${stage}-%j.err" \
  --export="ALL,DOCGRID_STAGE=${stage}" \
  "${resource_args[@]}" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/train_stage.sbatch"
