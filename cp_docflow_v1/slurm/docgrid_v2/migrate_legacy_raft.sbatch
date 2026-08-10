#!/usr/bin/env bash
#SBATCH --job-name=docgrid-migrate-raft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_LEGACY_CSV:?set DOCGRID_LEGACY_CSV}"
: "${DOCGRID_MIGRATED_MANIFEST:?set DOCGRID_MIGRATED_MANIFEST}"
: "${DOCGRID_MIGRATED_MAP_DIR:?set DOCGRID_MIGRATED_MAP_DIR}"
checkpoint="${DOCGRID_RAFT_CHECKPOINT:-${DOCGRID_PROJECT_ROOT}/../models/torch_cache/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"
generator="${DOCGRID_RAFT_GENERATOR:-${DOCGRID_PROJECT_ROOT}/../utils/generate_flow_labels.py}"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.migrate_legacy_raft
  --csv "${DOCGRID_LEGACY_CSV}"
  --output-manifest "${DOCGRID_MIGRATED_MANIFEST}"
  --derived-map-dir "${DOCGRID_MIGRATED_MAP_DIR}"
  --label-checkpoint "${checkpoint}"
  --generator-script "${generator}"
  --flow-source-height "${DOCGRID_LEGACY_FLOW_HEIGHT:-1024}"
  --flow-source-width "${DOCGRID_LEGACY_FLOW_WIDTH:-1024}"
)
if [[ -n "${DOCGRID_MIGRATION_LIMIT:-}" ]]; then
  command+=(--limit "${DOCGRID_MIGRATION_LIMIT}")
fi

echo "Migrating legacy labels as exploratory-only raft_pseudo data"
"${command[@]}"
