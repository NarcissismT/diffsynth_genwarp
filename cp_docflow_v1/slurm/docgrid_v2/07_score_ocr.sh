#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_OCR_TRANSCRIPTS:?set DOCGRID_OCR_TRANSCRIPTS to CSV or JSONL OCR text}"
: "${DOCGRID_OCR_OUTPUT:?set DOCGRID_OCR_OUTPUT to a new evidence directory}"
: "${DOCGRID_OCR_ENGINE:?set DOCGRID_OCR_ENGINE}"
: "${DOCGRID_OCR_ENGINE_VERSION:?set DOCGRID_OCR_ENGINE_VERSION}"
: "${DOCGRID_GEOMETRY_EVALUATION:?set DOCGRID_GEOMETRY_EVALUATION to metrics.json}"
: "${DOCGRID_GEOMETRY_PER_SAMPLE:?set DOCGRID_GEOMETRY_PER_SAMPLE to per_sample.csv}"
for required in \
  "${DOCGRID_OCR_TRANSCRIPTS}" \
  "${DOCGRID_GEOMETRY_EVALUATION}" \
  "${DOCGRID_GEOMETRY_PER_SAMPLE}"; do
  [[ -f "${required}" ]] || {
    echo "missing OCR scoring input: ${required}" >&2
    exit 2
  }
done

mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/score_ocr-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/score_ocr-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/score_ocr.sbatch"
