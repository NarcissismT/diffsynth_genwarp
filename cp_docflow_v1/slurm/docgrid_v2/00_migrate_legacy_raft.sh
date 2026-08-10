#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_LEGACY_CSV:?set DOCGRID_LEGACY_CSV}"
: "${DOCGRID_MIGRATED_MANIFEST:?set DOCGRID_MIGRATED_MANIFEST}"
: "${DOCGRID_MIGRATED_MAP_DIR:?set DOCGRID_MIGRATED_MAP_DIR}"
[[ -x "${DOCGRID_PYTHON}" ]] || {
  echo "DOCGRID_PYTHON is not executable: ${DOCGRID_PYTHON}" >&2
  exit 2
}
[[ -f "${DOCGRID_LEGACY_CSV}" ]] || {
  echo "missing legacy CSV: ${DOCGRID_LEGACY_CSV}" >&2
  exit 2
}

mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/migrate_legacy_raft-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/migrate_legacy_raft-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/migrate_legacy_raft.sbatch"
