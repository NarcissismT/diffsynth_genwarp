#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stage="${1:?usage: submit_multiseed.sh STAGE [extra sbatch arguments]}"
shift
seeds="${DOCGRID_SEEDS:-1337,2027,3407}"
IFS=',' read -r -a seed_values <<< "${seeds}"
if [[ "${#seed_values[@]}" -lt 3 ]]; then
  echo "multi-seed submission requires at least three comma-separated seeds" >&2
  exit 2
fi
for seed in "${seed_values[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "invalid seed: ${seed}" >&2; exit 2; }
  docgrid_preflight "${stage}" "${seed}"
done
mkdir -p "${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2"

last_index=$((${#seed_values[@]} - 1))
resource_args=(--mem="${DOCGRID_STANDARD_HOST_MEM:-64G}")
case "${stage}" in
  stage4_qwen|stage5_full_page)
    resource_args=(--mem="${DOCGRID_QWEN_HOST_MEM:-192G}")
    ;;
esac
exec sbatch \
  --array="0-${last_index}" \
  --job-name="docgrid-${stage}-3seed" \
  --output="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/${stage}-seed-%A_%a.out" \
  --error="${DOCGRID_PROJECT_ROOT}/slurm_logs/docgrid_v2/${stage}-seed-%A_%a.err" \
  --export="ALL,DOCGRID_STAGE=${stage},DOCGRID_SEEDS=${seeds}" \
  "${resource_args[@]}" \
  "$@" \
  "${DOCGRID_SLURM_DIR}/train_stage.sbatch"
