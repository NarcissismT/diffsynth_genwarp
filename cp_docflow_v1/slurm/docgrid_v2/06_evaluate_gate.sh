#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_GATE:?set DOCGRID_GATE=gate1..gate5}"
: "${DOCGRID_CHECKPOINT:?set DOCGRID_CHECKPOINT}"
: "${DOCGRID_EVAL_MANIFEST:?set DOCGRID_EVAL_MANIFEST}"
: "${DOCGRID_EVAL_OUTPUT:?set DOCGRID_EVAL_OUTPUT}"
mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --job-name="docgrid-${DOCGRID_GATE}" \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/${DOCGRID_GATE}-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/${DOCGRID_GATE}-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/evaluate_gate.sbatch"

