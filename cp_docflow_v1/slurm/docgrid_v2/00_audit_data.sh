#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/stage0_audit-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/stage0_audit-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/audit_stage0.sbatch"
