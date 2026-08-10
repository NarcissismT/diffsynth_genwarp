#!/usr/bin/env bash
# Full-page source audit followed by the canonical immutable analytic renderer.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_ANALYTIC_INPUT_CSV:?set DOCGRID_ANALYTIC_INPUT_CSV}"
: "${DOCGRID_ANALYTIC_SHARDS:?set DOCGRID_ANALYTIC_SHARDS}"
: "${SLURM_ARRAY_TASK_ID:?this job must run as a Slurm array}"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
validation_command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.validate_rectified_sources
  --input-csv "${DOCGRID_ANALYTIC_INPUT_CSV}"
  --image-column "${DOCGRID_ANALYTIC_IMAGE_COLUMN:-image}"
  --category-column "${DOCGRID_ANALYTIC_CATEGORY_COLUMN:-category}"
  --seed "${DOCGRID_RENDER_SEED:-1337}"
  --shard-index "${SLURM_ARRAY_TASK_ID}"
  --num-shards "${DOCGRID_ANALYTIC_SHARDS}"
  --target-height "${DOCGRID_RENDER_HEIGHT:-1024}"
  --target-width "${DOCGRID_RENDER_WIDTH:-768}"
  --minimum-source-height "${DOCGRID_FULL_PAGE_MIN_SOURCE_HEIGHT:-1024}"
  --minimum-source-width "${DOCGRID_FULL_PAGE_MIN_SOURCE_WIDTH:-768}"
  --aspect-tolerance "${DOCGRID_FULL_PAGE_ASPECT_TOLERANCE:-0.01}"
)
if [[ -n "${DOCGRID_ANALYTIC_MAX_DOCUMENTS:-}" ]]; then
  validation_command+=(--max-documents "${DOCGRID_ANALYTIC_MAX_DOCUMENTS}")
fi
"${validation_command[@]}"

# The source audit proves that this fixed resize is an isotropic downscale (up
# to the declared tolerance), so character and table-line aspect is preserved.
bash "${DOCGRID_SLURM_DIR}/render_analytic_gt_shard.sbatch"
