#!/usr/bin/env bash
# Portal resources: array 0..63, 1x A800 per task, 8 CPU, 64G host RAM.
# This is the independent 1024x768 Stage-5 corpus, not the 512x512 Stage-1 corpus.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

export DOCGRID_SHARD_INDEX="${DOCGRID_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
: "${DOCGRID_SHARD_INDEX:?set an array index in 0..63}"
[[ "${DOCGRID_SHARD_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "DOCGRID_SHARD_INDEX must be an integer" >&2
  exit 2
}

: "${DOCGRID_FULL_PAGE_INPUT_CSV:?set a CSV of flat full-resolution pages; the 512x512 metadata_with_flow.csv is forbidden}"
export DOCGRID_ANALYTIC_INPUT_CSV="${DOCGRID_FULL_PAGE_INPUT_CSV}"
export DOCGRID_ANALYTIC_IMAGE_COLUMN="${DOCGRID_ANALYTIC_IMAGE_COLUMN:-image}"
export DOCGRID_ANALYTIC_CATEGORY_COLUMN="${DOCGRID_ANALYTIC_CATEGORY_COLUMN:-category}"
export DOCGRID_ANALYTIC_SHARDS="${DOCGRID_ANALYTIC_SHARDS:-64}"
export DOCGRID_ANALYTIC_SHARD_ROOT="${DOCGRID_ANALYTIC_SHARD_ROOT:-${DOCGRID_DATA_ROOT}/analytic_full_page_shards_seed1337_1024x768}"
# One full-resolution warp per document keeps this corpus tractable. Override
# these two values explicitly before the first shard if a larger immutable run
# is intended; every shard must use identical values.
export DOCGRID_ANALYTIC_VARIANTS="${DOCGRID_ANALYTIC_VARIANTS:-1}"
export DOCGRID_ANALYTIC_MAX_DOCUMENTS="${DOCGRID_ANALYTIC_MAX_DOCUMENTS:-20000}"
export DOCGRID_RENDER_SEED="${DOCGRID_RENDER_SEED:-1337}"
export DOCGRID_RENDER_HEIGHT="${DOCGRID_RENDER_HEIGHT:-1024}"
export DOCGRID_RENDER_WIDTH="${DOCGRID_RENDER_WIDTH:-768}"
export DOCGRID_FULL_PAGE_MIN_SOURCE_HEIGHT="${DOCGRID_FULL_PAGE_MIN_SOURCE_HEIGHT:-1024}"
export DOCGRID_FULL_PAGE_MIN_SOURCE_WIDTH="${DOCGRID_FULL_PAGE_MIN_SOURCE_WIDTH:-768}"
export DOCGRID_FULL_PAGE_ASPECT_TOLERANCE="${DOCGRID_FULL_PAGE_ASPECT_TOLERANCE:-0.01}"
export DOCGRID_ANALYTIC_DEVICE="${DOCGRID_ANALYTIC_DEVICE:-cuda}"

((DOCGRID_SHARD_INDEX < DOCGRID_ANALYTIC_SHARDS)) || {
  echo "shard index ${DOCGRID_SHARD_INDEX} is outside 0..$((DOCGRID_ANALYTIC_SHARDS - 1))" >&2
  exit 2
}

docgrid_run_container 8 \
  'export SLURM_ARRAY_TASK_ID="${DOCGRID_SHARD_INDEX}"; bash cp_docflow_v1/slurm/docgrid_v2/render_full_page_gt_shard.sbatch'
