#!/usr/bin/env bash
# Runs inside the container after a human has reviewed immutable evaluation output.
set -euo pipefail

: "${DOCGRID_PYTHON:?DOCGRID_PYTHON is required}"
: "${DOCGRID_GATE:?DOCGRID_GATE=gate1..gate5 is required}"
: "${DOCGRID_EVAL_OUTPUT:?DOCGRID_EVAL_OUTPUT is required}"
: "${DOCGRID_GATE_RECEIPT:?DOCGRID_GATE_RECEIPT is required}"
: "${DOCGRID_GATE_REVIEWER:?DOCGRID_GATE_REVIEWER is required}"
: "${DOCGRID_GATE_REVIEW_NOTE:?DOCGRID_GATE_REVIEW_NOTE is required}"
: "${DOCGRID_GATE_DECISION:?DOCGRID_GATE_DECISION=passed|failed is required}"

case "${DOCGRID_GATE_DECISION}" in
  passed) decision=(--passed) ;;
  failed) decision=(--failed) ;;
  *) echo "DOCGRID_GATE_DECISION must be passed or failed" >&2; exit 2 ;;
esac

command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.gates
  --gate "${DOCGRID_GATE}"
  --evaluation "${DOCGRID_EVAL_OUTPUT}/metrics.json"
  --output "${DOCGRID_GATE_RECEIPT}"
  --reviewer "${DOCGRID_GATE_REVIEWER}"
  --review-note "${DOCGRID_GATE_REVIEW_NOTE}"
  "${decision[@]}"
)
if [[ -n "${DOCGRID_BASELINE_EVALUATION:-}" ]]; then
  command+=(--baseline-evaluation "${DOCGRID_BASELINE_EVALUATION}")
fi
if [[ -n "${DOCGRID_GATE_EVIDENCE:-}" ]]; then
  command+=(--evidence "${DOCGRID_GATE_EVIDENCE}")
fi
"${command[@]}"
