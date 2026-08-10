#!/usr/bin/env bash
#SBATCH --job-name=docgrid-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

gate="${DOCGRID_GATE:?DOCGRID_GATE=gate1..gate5 is required}"
checkpoint="${DOCGRID_CHECKPOINT:?DOCGRID_CHECKPOINT is required}"
manifest="${DOCGRID_EVAL_MANIFEST:?DOCGRID_EVAL_MANIFEST is required}"
output="${DOCGRID_EVAL_OUTPUT:?DOCGRID_EVAL_OUTPUT is required}"
case "${gate}" in
  gate1|gate2|gate3|gate4|gate5) ;;
  *) echo "invalid DOCGRID_GATE=${gate}" >&2; exit 2 ;;
esac

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.evaluate_full
  --checkpoint "${checkpoint}"
  --manifest "${manifest}"
  --output-dir "${output}"
  --device cuda
  --gate
  --allowed-label-provenance analytic_gt renderer_gt
)
if [[ "${gate}" == "gate5" ]]; then
  command+=(--export-ocr-images)
fi
"${command[@]}"

if [[ -n "${DOCGRID_GATE_DECISION:-}" ]]; then
  case "${DOCGRID_GATE_DECISION}" in
    passed) decision=(--passed) ;;
    failed) decision=(--failed) ;;
    *) echo "DOCGRID_GATE_DECISION must be passed or failed" >&2; exit 2 ;;
  esac
  receipt="${DOCGRID_GATE_RECEIPT:-runs/docgrid_v2/gates/${gate}.json}"
  command=(
    "${DOCGRID_PYTHON}" -m cp_docflow.gates
    --gate "${gate}"
    --evaluation "${output}/metrics.json"
    --output "${receipt}"
    --reviewer "${DOCGRID_GATE_REVIEWER:?reviewer is required for a receipt}"
    --review-note "${DOCGRID_GATE_REVIEW_NOTE:?review note is required for a receipt}"
    "${decision[@]}"
  )
  if [[ -n "${DOCGRID_BASELINE_EVALUATION:-}" ]]; then
    command+=(--baseline-evaluation "${DOCGRID_BASELINE_EVALUATION}")
  fi
  if [[ -n "${DOCGRID_GATE_EVIDENCE:-}" ]]; then
    command+=(--evidence "${DOCGRID_GATE_EVIDENCE}")
  fi
  "${command[@]}"
fi
