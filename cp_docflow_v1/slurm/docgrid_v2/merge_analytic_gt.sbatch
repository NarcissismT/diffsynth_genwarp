#!/usr/bin/env bash
#SBATCH --job-name=docgrid-merge-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_ANALYTIC_SHARD_ROOT:?set DOCGRID_ANALYTIC_SHARD_ROOT}"
: "${DOCGRID_ANALYTIC_MERGED_OUTPUT:?set DOCGRID_ANALYTIC_MERGED_OUTPUT}"
cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Merging immutable analytic GT shards from ${DOCGRID_ANALYTIC_SHARD_ROOT}"
"${DOCGRID_PYTHON}" -m cp_docflow.merge_analytic_shards \
  --shard-root "${DOCGRID_ANALYTIC_SHARD_ROOT}" \
  --output-dir "${DOCGRID_ANALYTIC_MERGED_OUTPUT}"
