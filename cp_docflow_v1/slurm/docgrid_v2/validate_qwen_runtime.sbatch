#!/usr/bin/env bash
#SBATCH --job-name=docgrid-qwen-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_QWEN_PROBE_IMAGE:?set DOCGRID_QWEN_PROBE_IMAGE}"
qwen_model="${DOCGRID_QWEN_MODEL:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit}"
report="${DOCGRID_QWEN_VALIDATION_REPORT:-${DOCGRID_PROJECT_ROOT}/runs/docgrid_v2/qwen_runtime_validation/report.json}"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.validate_qwen_runtime
  --image "${DOCGRID_QWEN_PROBE_IMAGE}"
  --model-id "${qwen_model}"
  --output "${report}"
  --device cuda
  --input-height 512
  --input-width 512
  --feature-type hidden
  --feature-layers -24 -12 -1
  --feature-dtype bfloat16
)
"${command[@]}"
