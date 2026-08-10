#!/usr/bin/env bash
#SBATCH --job-name=docgrid-prior-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

manifest="${DOCGRID_EVAL_MANIFEST:?DOCGRID_EVAL_MANIFEST is required}"
config="${DOCGRID_BASELINE_CONFIG:?DOCGRID_BASELINE_CONFIG is required}"
output="${DOCGRID_BASELINE_EVALUATION_OUTPUT:?DOCGRID_BASELINE_EVALUATION_OUTPUT is required}"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${DOCGRID_PYTHON}" -m cp_docflow.evaluate_frozen_prior \
  --config "${config}" \
  --manifest "${manifest}" \
  --output-dir "${output}" \
  --device cuda:0
