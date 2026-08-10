#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_ANALYTIC_INPUT_CSV:?set DOCGRID_ANALYTIC_INPUT_CSV to the flat-document CSV}"
[[ -x "${DOCGRID_PYTHON}" ]] || {
  echo "DOCGRID_PYTHON is not executable: ${DOCGRID_PYTHON}" >&2
  exit 2
}
[[ -f "${DOCGRID_ANALYTIC_INPUT_CSV}" ]] || {
  echo "missing flat-document CSV: ${DOCGRID_ANALYTIC_INPUT_CSV}" >&2
  exit 2
}

mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/render_analytic_gt-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/render_analytic_gt-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/render_analytic_gt.sbatch"
