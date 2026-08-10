#!/usr/bin/env bash
#SBATCH --job-name=docgrid-stage0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${DOCGRID_PROJECT_ROOT}"
export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${DOCGRID_PYTHON}" -m pytest -q \
  tests/test_geometry_contract.py \
  tests/test_data_contract.py

train_manifest="${DOCGRID_TRAIN_MANIFEST:-data/train_docgrid_v2.jsonl}"
val_manifest="${DOCGRID_VAL_MANIFEST:-data/val_docgrid_v2.jsonl}"
if [[ -n "${DOCGRID_AUDIT_OUTPUT:-}" ]]; then
  audit_output="${DOCGRID_AUDIT_OUTPUT}"
elif [[ -n "${DOCGRID_FROZEN_CONTRACT:-}" ]]; then
  audit_output="$(dirname "${DOCGRID_FROZEN_CONTRACT}")"
else
  audit_output="runs/docgrid_v2/stage0_audit"
fi
command=(
  "${DOCGRID_PYTHON}" -m cp_docflow.audit_data
  --train-manifest "${train_manifest}"
  --val-manifest "${val_manifest}"
  --output-dir "${audit_output}"
  --allowed-label-provenance analytic_gt renderer_gt
  --seeds 1337 2027 3407
)
if [[ -n "${DOCGRID_TEST_MANIFEST:-}" ]]; then
  command+=(--test-manifest "${DOCGRID_TEST_MANIFEST}")
fi
baseline_count=0
for value in \
  "${DOCGRID_BASELINE_CHECKPOINT:-}" \
  "${DOCGRID_BASELINE_CONFIG:-}" \
  "${DOCGRID_BASELINE_METRICS:-}"; do
  [[ -n "${value}" ]] && baseline_count=$((baseline_count + 1))
done
if [[ "${baseline_count}" -ne 0 && "${baseline_count}" -ne 3 ]]; then
  echo "baseline checkpoint/config/metrics must all be set together" >&2
  exit 2
fi
if [[ "${baseline_count}" -eq 3 ]]; then
  command+=(
    --baseline-checkpoint "${DOCGRID_BASELINE_CHECKPOINT}"
    --baseline-config "${DOCGRID_BASELINE_CONFIG}"
    --baseline-metrics "${DOCGRID_BASELINE_METRICS}"
  )
fi
"${command[@]}"
