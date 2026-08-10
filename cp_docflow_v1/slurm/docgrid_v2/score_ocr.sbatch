#!/usr/bin/env bash
#SBATCH --job-name=docgrid-score-ocr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_OCR_TRANSCRIPTS:?set DOCGRID_OCR_TRANSCRIPTS}"
: "${DOCGRID_OCR_OUTPUT:?set DOCGRID_OCR_OUTPUT}"
: "${DOCGRID_OCR_ENGINE:?set DOCGRID_OCR_ENGINE}"
: "${DOCGRID_OCR_ENGINE_VERSION:?set DOCGRID_OCR_ENGINE_VERSION}"
: "${DOCGRID_GEOMETRY_EVALUATION:?set DOCGRID_GEOMETRY_EVALUATION}"
: "${DOCGRID_GEOMETRY_PER_SAMPLE:?set DOCGRID_GEOMETRY_PER_SAMPLE}"
: "${DOCGRID_OCR_IMAGE_MANIFEST:?set DOCGRID_OCR_IMAGE_MANIFEST}"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${DOCGRID_PYTHON}" -m cp_docflow.evaluation.ocr_metrics \
  --transcripts "${DOCGRID_OCR_TRANSCRIPTS}" \
  --output-dir "${DOCGRID_OCR_OUTPUT}" \
  --ocr-engine "${DOCGRID_OCR_ENGINE}" \
  --ocr-engine-version "${DOCGRID_OCR_ENGINE_VERSION}" \
  --geometry-evaluation "${DOCGRID_GEOMETRY_EVALUATION}" \
  --geometry-per-sample "${DOCGRID_GEOMETRY_PER_SAMPLE}" \
  --ocr-image-manifest "${DOCGRID_OCR_IMAGE_MANIFEST}"
