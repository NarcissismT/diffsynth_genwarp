#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_ANALYTIC_INPUT_CSV:?set DOCGRID_ANALYTIC_INPUT_CSV to the flat-document CSV}"
: "${DOCGRID_ANALYTIC_SHARDS:?set DOCGRID_ANALYTIC_SHARDS to the Slurm array size}"
[[ -x "${DOCGRID_PYTHON}" ]] || {
  echo "DOCGRID_PYTHON is not executable: ${DOCGRID_PYTHON}" >&2
  exit 2
}
[[ -f "${DOCGRID_ANALYTIC_INPUT_CSV}" ]] || {
  echo "missing flat-document CSV: ${DOCGRID_ANALYTIC_INPUT_CSV}" >&2
  exit 2
}
[[ "${DOCGRID_ANALYTIC_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "DOCGRID_ANALYTIC_SHARDS must be a positive integer" >&2
  exit 2
}

export DOCGRID_ANALYTIC_SHARD_ROOT="${DOCGRID_ANALYTIC_SHARD_ROOT:-${DOCGRID_PROJECT_ROOT}/data/docgrid_v2_analytic_shards}"
export DOCGRID_ANALYTIC_SHARDS
mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --array="0-$((DOCGRID_ANALYTIC_SHARDS - 1))" \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/render_analytic_gt-%A_%a.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/render_analytic_gt-%A_%a.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/render_analytic_gt_shard.sbatch"
