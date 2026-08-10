#!/usr/bin/env bash
# Portal resources: CPU job, 8 CPU, 32G host RAM. Requires fixed OCR transcripts.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

: "${DOCGRID_OCR_TRANSCRIPTS:?export DOCGRID_OCR_TRANSCRIPTS before submission}"
: "${DOCGRID_OCR_ENGINE:?export the fixed DOCGRID_OCR_ENGINE before submission}"
: "${DOCGRID_OCR_ENGINE_VERSION:?export the exact DOCGRID_OCR_ENGINE_VERSION before submission}"
export DOCGRID_SEED="${DOCGRID_SEED:-1337}"
readonly gate5_eval="${DOCGRID_RUN_ROOT}/stage5_full_page/seed-${DOCGRID_SEED}/gate5_eval"
export DOCGRID_GEOMETRY_EVALUATION="${DOCGRID_GEOMETRY_EVALUATION:-${gate5_eval}/metrics.json}"
export DOCGRID_GEOMETRY_PER_SAMPLE="${DOCGRID_GEOMETRY_PER_SAMPLE:-${gate5_eval}/per_sample.csv}"
export DOCGRID_OCR_IMAGE_MANIFEST="${DOCGRID_OCR_IMAGE_MANIFEST:-${gate5_eval}/ocr_images.jsonl}"
export DOCGRID_OCR_OUTPUT="${DOCGRID_OCR_OUTPUT:-${DOCGRID_RUN_ROOT}/stage5_full_page/ocr_evidence}"

docgrid_run_container 8 \
  'bash cp_docflow_v1/slurm/docgrid_v2/score_ocr.sbatch'
