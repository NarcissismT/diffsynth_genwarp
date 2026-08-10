#!/usr/bin/env bash
#SBATCH --job-name=docgrid-full-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

checkpoint="${DOCGRID_CHECKPOINT:?DOCGRID_CHECKPOINT is required}"
manifest="${DOCGRID_EVAL_MANIFEST:?DOCGRID_EVAL_MANIFEST is required}"
output="${DOCGRID_BASELINE_EVALUATION_OUTPUT:?DOCGRID_BASELINE_EVALUATION_OUTPUT is required}"
input_height="${DOCGRID_EVAL_INPUT_HEIGHT:-1024}"
input_width="${DOCGRID_EVAL_INPUT_WIDTH:-768}"
output_height="${DOCGRID_EVAL_OUTPUT_HEIGHT:-1024}"
output_width="${DOCGRID_EVAL_OUTPUT_WIDTH:-768}"

docgrid_require_file "${checkpoint}" "Stage-2 deterministic checkpoint"
docgrid_require_file "${manifest}" "full-page validation manifest"
docgrid_require_file "${DOCGRID_STAGE5_FROZEN_CONTRACT:?DOCGRID_STAGE5_FROZEN_CONTRACT is required}" \
  "full-page frozen data contract"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${DOCGRID_PYTHON}" -m cp_docflow.evaluate_full \
  --checkpoint "${checkpoint}" \
  --manifest "${manifest}" \
  --output-dir "${output}" \
  --device cuda \
  --allowed-label-provenance analytic_gt renderer_gt \
  --input-work-size "${input_height}" "${input_width}" \
  --output-work-size "${output_height}" "${output_width}"
