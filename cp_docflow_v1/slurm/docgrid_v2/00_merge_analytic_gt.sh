#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_ANALYTIC_SHARD_ROOT:?set DOCGRID_ANALYTIC_SHARD_ROOT to the completed shard directory}"
[[ -x "${DOCGRID_PYTHON}" ]] || {
  echo "DOCGRID_PYTHON is not executable: ${DOCGRID_PYTHON}" >&2
  exit 2
}
[[ -d "${DOCGRID_ANALYTIC_SHARD_ROOT}" ]] || {
  echo "missing analytic shard directory: ${DOCGRID_ANALYTIC_SHARD_ROOT}" >&2
  exit 2
}

export DOCGRID_ANALYTIC_MERGED_OUTPUT="${DOCGRID_ANALYTIC_MERGED_OUTPUT:-${DOCGRID_PROJECT_ROOT}/data/docgrid_v2_analytic_merged}"
mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/merge_analytic_gt-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/merge_analytic_gt-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/merge_analytic_gt.sbatch"
