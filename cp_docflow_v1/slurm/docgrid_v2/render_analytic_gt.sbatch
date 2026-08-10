#!/usr/bin/env bash
#SBATCH --job-name=docgrid-render-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_ANALYTIC_INPUT_CSV:?set DOCGRID_ANALYTIC_INPUT_CSV}"
output_dir="${DOCGRID_ANALYTIC_OUTPUT:-${DOCGRID_PROJECT_ROOT}/data/docgrid_v2_analytic}"
cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.render_analytic_gt
  --input-csv "${DOCGRID_ANALYTIC_INPUT_CSV}"
  --output-dir "${output_dir}"
  --image-column "${DOCGRID_ANALYTIC_IMAGE_COLUMN:-image}"
  --category-column "${DOCGRID_ANALYTIC_CATEGORY_COLUMN:-category}"
  --variants-per-document "${DOCGRID_ANALYTIC_VARIANTS:-3}"
  --seed "${DOCGRID_RENDER_SEED:-1337}"
  --train-ratio "${DOCGRID_TRAIN_RATIO:-0.90}"
  --val-ratio "${DOCGRID_VAL_RATIO:-0.05}"
  --test-ratio "${DOCGRID_TEST_RATIO:-0.05}"
  --output-height "${DOCGRID_RENDER_HEIGHT:-512}"
  --output-width "${DOCGRID_RENDER_WIDTH:-512}"
  --device "${DOCGRID_ANALYTIC_DEVICE:-cuda}"
)
if [[ -n "${DOCGRID_ANALYTIC_MAX_DOCUMENTS:-}" ]]; then
  command+=(--max-documents "${DOCGRID_ANALYTIC_MAX_DOCUMENTS}")
fi

echo "Rendering exact analytic GT to ${output_dir}"
"${command[@]}"
