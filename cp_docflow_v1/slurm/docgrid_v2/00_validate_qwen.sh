#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${DOCGRID_QWEN_PROBE_IMAGE:?set DOCGRID_QWEN_PROBE_IMAGE to one representative warped page}"
[[ -x "${DOCGRID_PYTHON}" ]] || {
  echo "DOCGRID_PYTHON is not executable: ${DOCGRID_PYTHON}" >&2
  exit 2
}
[[ -f "${DOCGRID_QWEN_PROBE_IMAGE}" ]] || {
  echo "missing Qwen probe image: ${DOCGRID_QWEN_PROBE_IMAGE}" >&2
  exit 2
}

mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"
exec sbatch \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/validate_qwen-%j.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/validate_qwen-%j.err" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/validate_qwen_runtime.sbatch"
