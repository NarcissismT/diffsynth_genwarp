#!/usr/bin/env bash
# Shared host-side launcher for the IntSig srun container environment.
set -euo pipefail

readonly DOCGRID_CONTAINER_JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DOCGRID_CONTAINER_PROJECT_ROOT="$(cd "${DOCGRID_CONTAINER_JOBS_DIR}/../../.." && pwd)"

export DOCGRID_CONTAINER_IMAGE="${DOCGRID_CONTAINER_IMAGE:-docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
export DOCGRID_CONTAINER_WORKDIR="${DOCGRID_CONTAINER_WORKDIR:-/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp}"
export DOCGRID_DATA_ROOT="${DOCGRID_DATA_ROOT:-/juicefs-algorithm/data/IPT/zhuochu_yang/docgrid_v2}"
export DOCGRID_RUN_ROOT="${DOCGRID_RUN_ROOT:-${DOCGRID_DATA_ROOT}/runs}"

export HF_HOME="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"
readonly docgrid_job_key="${SLURM_JOB_ID:-manual}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/slurm_${docgrid_job_key}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/slurm_${docgrid_job_key}/torch_extensions}"

docgrid_container_env_csv() {
  local -a names=(
    HF_HOME TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR
    DOCGRID_DATA_ROOT DOCGRID_RUN_ROOT DOCGRID_SHARD_INDEX
    DOCGRID_ANALYTIC_INPUT_CSV DOCGRID_ANALYTIC_IMAGE_COLUMN
    DOCGRID_ANALYTIC_CATEGORY_COLUMN DOCGRID_ANALYTIC_VARIANTS
    DOCGRID_ANALYTIC_MAX_DOCUMENTS
    DOCGRID_FULL_PAGE_INPUT_CSV DOCGRID_FULL_PAGE_MIN_SOURCE_HEIGHT
    DOCGRID_FULL_PAGE_MIN_SOURCE_WIDTH DOCGRID_FULL_PAGE_ASPECT_TOLERANCE
    DOCGRID_RENDER_SEED DOCGRID_RENDER_HEIGHT DOCGRID_RENDER_WIDTH
    DOCGRID_ANALYTIC_DEVICE DOCGRID_ANALYTIC_SHARDS
    DOCGRID_ANALYTIC_SHARD_ROOT DOCGRID_ANALYTIC_MERGED_OUTPUT
    DOCGRID_TRAIN_MANIFEST DOCGRID_VAL_MANIFEST DOCGRID_TEST_MANIFEST
    DOCGRID_AUDIT_OUTPUT DOCGRID_FROZEN_CONTRACT
    DOCGRID_BASELINE_CHECKPOINT DOCGRID_BASELINE_CONFIG
    DOCGRID_BASELINE_METRICS DOCGRID_BASELINE_EVALUATION_OUTPUT
    DOCGRID_QWEN_MODEL DOCGRID_QWEN_VALIDATION_REPORT
    DOCGRID_QWEN_PROBE_IMAGE DOCGRID_STAGE DOCGRID_SEED
    DOCGRID_OUTPUT_DIR DOCGRID_PARENT_CHECKPOINT
    DOCGRID_GATE1_RECEIPT DOCGRID_GATE2_RECEIPT
    DOCGRID_GATE3_RECEIPT DOCGRID_GATE4_RECEIPT
    DOCGRID_STAGE5_TRAIN_MANIFEST DOCGRID_STAGE5_VAL_MANIFEST
    DOCGRID_STAGE5_TEST_MANIFEST
    DOCGRID_STAGE5_FROZEN_CONTRACT DOCGRID_GATE DOCGRID_CHECKPOINT
    DOCGRID_EVAL_MANIFEST DOCGRID_EVAL_OUTPUT
    DOCGRID_EVAL_INPUT_HEIGHT DOCGRID_EVAL_INPUT_WIDTH
    DOCGRID_EVAL_OUTPUT_HEIGHT DOCGRID_EVAL_OUTPUT_WIDTH
    DOCGRID_GATE_RECEIPT DOCGRID_GATE_REVIEWER DOCGRID_GATE_REVIEW_NOTE
    DOCGRID_GATE_DECISION DOCGRID_BASELINE_EVALUATION DOCGRID_GATE_EVIDENCE
    DOCGRID_OCR_TRANSCRIPTS DOCGRID_OCR_OUTPUT DOCGRID_OCR_ENGINE
    DOCGRID_OCR_ENGINE_VERSION DOCGRID_GEOMETRY_EVALUATION
    DOCGRID_GEOMETRY_PER_SAMPLE DOCGRID_OCR_IMAGE_MANIFEST
  )
  local -a available=()
  local name
  for name in "${names[@]}"; do
    [[ -v "${name}" ]] && available+=("${name}")
  done
  local IFS=,
  printf '%s\n' "${available[*]}"
}

docgrid_run_container() {
  local cpus="${1:?CPU count is required}"
  local worker_command="${2:?worker command is required}"
  local bootstrap
  bootstrap="set -euo pipefail; \
      export DOCGRID_PYTHON=\"\${DOCGRID_PYTHON:-\$(command -v python)}\"; \
      export PYTHONPATH=\"${DOCGRID_CONTAINER_WORKDIR}/cp_docflow_v1/src\"; \
      \"\${DOCGRID_PYTHON}\" -c 'import torch; print(\"python/cuda\", torch.__version__, torch.cuda.is_available())'; \
      ${worker_command}"

  # Some portals submit this host launcher directly; others already wrap the
  # selected script in ``srun --container-image ...``.  In the second layout,
  # recursively invoking srun fails because the container intentionally has no
  # Slurm client.  Execute the canonical worker in the current allocation.
  if ! command -v srun >/dev/null 2>&1; then
    if [[ -z "${SLURM_JOB_ID:-}" && "${DOCGRID_CONTAINER_ACTIVE:-0}" != "1" ]]; then
      echo "srun is unavailable outside an allocated container" >&2
      echo "submit this script through the Slurm platform or set DOCGRID_CONTAINER_ACTIVE=1 inside its container" >&2
      return 2
    fi
    echo "Detected an existing Slurm container; running the canonical worker without nested srun"
    bash -lc "${bootstrap}"
    return
  fi
  local container_env
  container_env="$(docgrid_container_env_csv)"
  srun --cpus-per-task="${cpus}" -K \
    --container-image="${DOCGRID_CONTAINER_IMAGE}" \
    --container-mounts=/juicefs-algorithm:/juicefs-algorithm \
    --container-workdir="${DOCGRID_CONTAINER_WORKDIR}" \
    --container-env="${container_env}" \
    bash -lc "${bootstrap}"
}
