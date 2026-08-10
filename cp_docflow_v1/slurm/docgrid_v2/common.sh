#!/usr/bin/env bash
set -euo pipefail

readonly DOCGRID_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DOCGRID_PROJECT_ROOT="$(cd "${DOCGRID_SLURM_DIR}/../.." && pwd)"
readonly DOCGRID_PYTHON="${DOCGRID_PYTHON:-/usr/bin/python}"

declare -Ar DOCGRID_CONFIGS=(
  [stage1_coarse]="configs/docgrid_v2/stage1_coarse.yaml"
  [stage2_warr]="configs/docgrid_v2/stage2_warr.yaml"
  [stage3_coordinate_fm]="configs/docgrid_v2/stage3_coordinate_fm.yaml"
  [stage4_qwen]="configs/docgrid_v2/stage4_qwen.yaml"
  [stage5_full_page]="configs/docgrid_v2/stage5_full_page.yaml"
)

docgrid_validate_stage() {
  local stage="${1:?stage is required}"
  if [[ -z "${DOCGRID_CONFIGS[$stage]+configured}" ]]; then
    echo "unknown stage: ${stage}" >&2
    return 2
  fi
  [[ -x "${DOCGRID_PYTHON}" ]] || {
    echo "DOCGRID_PYTHON is not executable: ${DOCGRID_PYTHON}" >&2
    return 2
  }
  [[ -f "${DOCGRID_PROJECT_ROOT}/${DOCGRID_CONFIGS[$stage]}" ]] || {
    echo "missing config: ${DOCGRID_CONFIGS[$stage]}" >&2
    return 2
  }
}

docgrid_require_file() {
  local path="${1:?path is required}"
  local role="${2:?role is required}"
  local resolved_path
  if [[ "${path}" = /* ]]; then
    resolved_path="${path}"
  else
    resolved_path="${DOCGRID_PROJECT_ROOT}/${path}"
  fi
  [[ -f "${resolved_path}" ]] || {
    echo "missing ${role}: ${resolved_path}" >&2
    return 2
  }
}

docgrid_frozen_contract_path() {
  local stage="${1:?stage is required}"
  if [[ "${stage}" == "stage5_full_page" ]]; then
    printf '%s\n' "${DOCGRID_STAGE5_FROZEN_CONTRACT:-runs/docgrid_v2/stage0_full_page_audit/frozen_contract.json}"
  else
    printf '%s\n' "${DOCGRID_FROZEN_CONTRACT:-runs/docgrid_v2/stage0_audit/frozen_contract.json}"
  fi
}

docgrid_gate_receipt_path() {
  local gate="${1:?gate is required}"
  case "${gate}" in
    gate1) printf '%s\n' "${DOCGRID_GATE1_RECEIPT:-runs/docgrid_v2/gates/gate1.json}" ;;
    gate2) printf '%s\n' "${DOCGRID_GATE2_RECEIPT:-runs/docgrid_v2/gates/gate2.json}" ;;
    gate3) printf '%s\n' "${DOCGRID_GATE3_RECEIPT:-runs/docgrid_v2/gates/gate3.json}" ;;
    gate4) printf '%s\n' "${DOCGRID_GATE4_RECEIPT:-runs/docgrid_v2/gates/gate4.json}" ;;
    *) echo "unknown Gate receipt: ${gate}" >&2; return 2 ;;
  esac
}

docgrid_parent_checkpoint_path() {
  local stage="${1:?stage is required}"
  local seed="${2:?seed is required}"
  if [[ -n "${DOCGRID_PARENT_CHECKPOINT:-}" ]]; then
    printf '%s\n' "${DOCGRID_PARENT_CHECKPOINT}"
    return
  fi
  case "${stage}" in
    stage2_warr)
      printf '%s\n' "${DOCGRID_STAGE2_PARENT_CHECKPOINT:-runs/docgrid_v2/stage1_coarse/seed-${seed}/best.pt}"
      ;;
    stage3_coordinate_fm)
      printf '%s\n' "${DOCGRID_STAGE3_PARENT_CHECKPOINT:-runs/docgrid_v2/stage2_warr/seed-${seed}/best.pt}"
      ;;
    stage4_qwen)
      printf '%s\n' "${DOCGRID_STAGE4_PARENT_CHECKPOINT:-runs/docgrid_v2/stage3_coordinate_fm/seed-${seed}/best.pt}"
      ;;
    stage5_full_page)
      printf '%s\n' "${DOCGRID_STAGE5_PARENT_CHECKPOINT:-runs/docgrid_v2/stage4_qwen/seed-${seed}/best.pt}"
      ;;
    *) echo "stage ${stage} has no parent checkpoint" >&2; return 2 ;;
  esac
}

docgrid_require_qwen() {
  local qwen_model="${DOCGRID_QWEN_MODEL:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit}"
  local validation_report="${DOCGRID_QWEN_VALIDATION_REPORT:-${DOCGRID_PROJECT_ROOT}/runs/docgrid_v2/qwen_runtime_validation/report.json}"
  [[ -d "${qwen_model}" ]] || {
    echo "missing local Qwen model: ${qwen_model}" >&2
    return 2
  }
  [[ -f "${qwen_model}/model_index.json" ]] || {
    echo "Qwen path is not a complete diffusers pipeline: ${qwen_model}" >&2
    return 2
  }
  grep -q 'QwenImageEditPipeline' "${qwen_model}/model_index.json" || {
    echo "unexpected Qwen pipeline class in ${qwen_model}/model_index.json" >&2
    return 2
  }
  "${DOCGRID_PYTHON}" -c \
    'from diffusers import QwenImageEditPipeline' >/dev/null 2>&1 || {
    echo "DOCGRID_PYTHON cannot import diffusers.QwenImageEditPipeline" >&2
    echo "install the project Qwen extras in the Slurm environment" >&2
    return 2
  }
  [[ -f "${validation_report}" ]] || {
    echo "missing real-Qwen runtime validation report: ${validation_report}" >&2
    echo "run slurm/docgrid_v2/00_validate_qwen.sh before Stage 4" >&2
    return 2
  }
  PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${DOCGRID_PYTHON}" -m cp_docflow.validate_qwen_runtime \
      --model-id "${qwen_model}" \
      --input-height 512 \
      --input-width 512 \
      --feature-type hidden \
      --feature-layers -24 -12 -1 \
      --feature-dtype bfloat16 \
      --check-report "${validation_report}" >/dev/null || {
        echo "Qwen runtime validation report is stale or incompatible" >&2
        return 2
      }
}

docgrid_preflight() {
  local stage="${1:?stage is required}"
  local seed="${2:-1337}"
  local frozen_contract
  local frozen_contract_role="Stage-0 frozen data contract"
  docgrid_validate_stage "${stage}"
  frozen_contract="$(docgrid_frozen_contract_path "${stage}")"
  if [[ "${stage}" == "stage5_full_page" ]]; then
    frozen_contract_role="full-page Stage-0 frozen data contract"
  fi
  docgrid_require_file "${frozen_contract}" "${frozen_contract_role}"
  case "${stage}" in
    stage1_coarse)
      ;;
    stage2_warr)
      docgrid_require_file "$(docgrid_parent_checkpoint_path "${stage}" "${seed}")" "Stage-1 checkpoint"
      docgrid_require_file "$(docgrid_gate_receipt_path gate1)" "Gate-1 receipt"
      ;;
    stage3_coordinate_fm)
      docgrid_require_file "$(docgrid_parent_checkpoint_path "${stage}" "${seed}")" "Stage-2 checkpoint"
      docgrid_require_file "$(docgrid_gate_receipt_path gate1)" "Gate-1 receipt"
      docgrid_require_file "$(docgrid_gate_receipt_path gate2)" "Gate-2 receipt"
      ;;
    stage4_qwen)
      docgrid_require_file "$(docgrid_parent_checkpoint_path "${stage}" "${seed}")" "Stage-3 checkpoint"
      docgrid_require_file "$(docgrid_gate_receipt_path gate1)" "Gate-1 receipt"
      docgrid_require_file "$(docgrid_gate_receipt_path gate2)" "Gate-2 receipt"
      docgrid_require_file "$(docgrid_gate_receipt_path gate3)" "Gate-3 receipt"
      docgrid_require_qwen
      ;;
    stage5_full_page)
      docgrid_require_file "$(docgrid_parent_checkpoint_path "${stage}" "${seed}")" "Stage-4 checkpoint"
      for gate in gate1 gate2 gate3 gate4; do
        docgrid_require_file "$(docgrid_gate_receipt_path "${gate}")" "${gate} receipt"
      done
      docgrid_require_qwen
      ;;
  esac
}

docgrid_run_stage() {
  local stage="${1:?stage is required}"
  local seed="${DOCGRID_SEED:-1337}"
  docgrid_preflight "${stage}" "${seed}"
  cd "${DOCGRID_PROJECT_ROOT}"
  export PYTHONPATH="${DOCGRID_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  declare -A stage_modules=(
    [stage1_coarse]="cp_docflow.training.train_deterministic"
    [stage2_warr]="cp_docflow.training.train_warr"
    [stage3_coordinate_fm]="cp_docflow.training.train_coordinate_fm"
    [stage4_qwen]="cp_docflow.training.train_qwen_condition"
    [stage5_full_page]="cp_docflow.training.finetune_full_page"
  )
  local module="${stage_modules[$stage]}"
  local command=(
    "${DOCGRID_PYTHON}" -m "${module}"
    --config "${DOCGRID_CONFIGS[$stage]}"
    --seed "${seed}"
  )
  if [[ "${stage}" != "stage1_coarse" ]]; then
    local parent
    parent="$(docgrid_parent_checkpoint_path "${stage}" "${seed}")"
    command+=(--parent-checkpoint "${parent}")
  fi
  if [[ -n "${DOCGRID_RESUME:-}" ]]; then
    command+=(--resume "${DOCGRID_RESUME}")
  fi
  if [[ -n "${DOCGRID_OUTPUT_DIR:-}" ]]; then
    command+=(--output-dir "${DOCGRID_OUTPUT_DIR}")
  fi
  echo "DocGrid-Flow stage=${stage} seed=${seed} config=${DOCGRID_CONFIGS[$stage]} python=${DOCGRID_PYTHON}"
  "${command[@]}"
}
