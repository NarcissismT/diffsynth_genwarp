#!/usr/bin/env bash
# Portal resources: Slurm array 0-63, 1x A800 per task, 8 CPU, 64G host RAM.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

export DOCGRID_SHARD_INDEX="${DOCGRID_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
: "${DOCGRID_SHARD_INDEX:?set an array index in 0..63}"
[[ "${DOCGRID_SHARD_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "DOCGRID_SHARD_INDEX must be an integer" >&2
  exit 2
}

export DOCGRID_ANALYTIC_INPUT_CSV="${DOCGRID_ANALYTIC_INPUT_CSV:-/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_flow.csv}"
export DOCGRID_ANALYTIC_IMAGE_COLUMN="${DOCGRID_ANALYTIC_IMAGE_COLUMN:-image}"
export DOCGRID_ANALYTIC_CATEGORY_COLUMN="${DOCGRID_ANALYTIC_CATEGORY_COLUMN:-category}"
export DOCGRID_ANALYTIC_SHARDS="${DOCGRID_ANALYTIC_SHARDS:-64}"
export DOCGRID_ANALYTIC_SHARD_ROOT="${DOCGRID_ANALYTIC_SHARD_ROOT:-${DOCGRID_DATA_ROOT}/analytic_shards_seed1337_512}"
export DOCGRID_ANALYTIC_VARIANTS="${DOCGRID_ANALYTIC_VARIANTS:-3}"
export DOCGRID_RENDER_SEED="${DOCGRID_RENDER_SEED:-1337}"
export DOCGRID_RENDER_HEIGHT="${DOCGRID_RENDER_HEIGHT:-512}"
export DOCGRID_RENDER_WIDTH="${DOCGRID_RENDER_WIDTH:-512}"
export DOCGRID_ANALYTIC_DEVICE="${DOCGRID_ANALYTIC_DEVICE:-cuda}"

((DOCGRID_SHARD_INDEX < DOCGRID_ANALYTIC_SHARDS)) || {
  echo "shard index ${DOCGRID_SHARD_INDEX} is outside 0..$((DOCGRID_ANALYTIC_SHARDS - 1))" >&2
  exit 2
}

docgrid_run_container 8 \
  'export SLURM_ARRAY_TASK_ID="${DOCGRID_SHARD_INDEX}"; bash cp_docflow_v1/slurm/docgrid_v2/render_analytic_gt_shard.sbatch'
