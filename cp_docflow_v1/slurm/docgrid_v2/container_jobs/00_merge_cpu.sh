#!/usr/bin/env bash
# Portal resources: CPU job, 8 CPU, 32G host RAM. Run after every shard succeeds.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

export DOCGRID_ANALYTIC_SHARD_ROOT="${DOCGRID_ANALYTIC_SHARD_ROOT:-${DOCGRID_DATA_ROOT}/analytic_shards_seed1337_512}"
export DOCGRID_ANALYTIC_MERGED_OUTPUT="${DOCGRID_ANALYTIC_MERGED_OUTPUT:-${DOCGRID_DATA_ROOT}/analytic_merged_seed1337_512}"

docgrid_run_container 8 \
  'bash cp_docflow_v1/slurm/docgrid_v2/merge_analytic_gt.sbatch'
