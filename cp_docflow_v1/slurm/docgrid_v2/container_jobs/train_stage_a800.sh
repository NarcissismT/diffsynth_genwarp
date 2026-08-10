#!/usr/bin/env bash
# Host-side worker used by 01..05 wrappers; do not submit this file without a stage.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

stage="${1:?usage: train_stage_a800.sh STAGE}"
export DOCGRID_STAGE="${stage}"
export DOCGRID_SEED="${DOCGRID_SEED:-1337}"

readonly merged_root="${DOCGRID_ANALYTIC_MERGED_OUTPUT:-${DOCGRID_DATA_ROOT}/analytic_merged_seed1337_512}"
export DOCGRID_TRAIN_MANIFEST="${DOCGRID_TRAIN_MANIFEST:-${merged_root}/manifests/train.jsonl}"
export DOCGRID_VAL_MANIFEST="${DOCGRID_VAL_MANIFEST:-${merged_root}/manifests/val.jsonl}"
export DOCGRID_TEST_MANIFEST="${DOCGRID_TEST_MANIFEST:-${merged_root}/manifests/test.jsonl}"
export DOCGRID_FROZEN_CONTRACT="${DOCGRID_FROZEN_CONTRACT:-${DOCGRID_RUN_ROOT}/stage0_audit/frozen_contract.json}"

export DOCGRID_GATE1_RECEIPT="${DOCGRID_GATE1_RECEIPT:-${DOCGRID_RUN_ROOT}/gates/gate1.json}"
export DOCGRID_GATE2_RECEIPT="${DOCGRID_GATE2_RECEIPT:-${DOCGRID_RUN_ROOT}/gates/gate2.json}"
export DOCGRID_GATE3_RECEIPT="${DOCGRID_GATE3_RECEIPT:-${DOCGRID_RUN_ROOT}/gates/gate3.json}"
export DOCGRID_GATE4_RECEIPT="${DOCGRID_GATE4_RECEIPT:-${DOCGRID_RUN_ROOT}/gates/gate4.json}"
export DOCGRID_QWEN_MODEL="${DOCGRID_QWEN_MODEL:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit}"
export DOCGRID_QWEN_VALIDATION_REPORT="${DOCGRID_QWEN_VALIDATION_REPORT:-${DOCGRID_RUN_ROOT}/qwen_runtime_validation/report.json}"

case "${stage}" in
  stage1_coarse)
    export DOCGRID_OUTPUT_DIR="${DOCGRID_OUTPUT_DIR:-${DOCGRID_RUN_ROOT}/stage1_coarse/seed-${DOCGRID_SEED}}"
    unset DOCGRID_PARENT_CHECKPOINT || true
    ;;
  stage2_warr)
    export DOCGRID_PARENT_CHECKPOINT="${DOCGRID_PARENT_CHECKPOINT:-${DOCGRID_RUN_ROOT}/stage1_coarse/seed-${DOCGRID_SEED}/best.pt}"
    export DOCGRID_OUTPUT_DIR="${DOCGRID_OUTPUT_DIR:-${DOCGRID_RUN_ROOT}/stage2_warr/seed-${DOCGRID_SEED}}"
    ;;
  stage3_coordinate_fm)
    export DOCGRID_PARENT_CHECKPOINT="${DOCGRID_PARENT_CHECKPOINT:-${DOCGRID_RUN_ROOT}/stage2_warr/seed-${DOCGRID_SEED}/best.pt}"
    export DOCGRID_OUTPUT_DIR="${DOCGRID_OUTPUT_DIR:-${DOCGRID_RUN_ROOT}/stage3_coordinate_fm/seed-${DOCGRID_SEED}}"
    ;;
  stage4_qwen)
    export DOCGRID_PARENT_CHECKPOINT="${DOCGRID_PARENT_CHECKPOINT:-${DOCGRID_RUN_ROOT}/stage3_coordinate_fm/seed-${DOCGRID_SEED}/best.pt}"
    export DOCGRID_OUTPUT_DIR="${DOCGRID_OUTPUT_DIR:-${DOCGRID_RUN_ROOT}/stage4_qwen/seed-${DOCGRID_SEED}}"
    ;;
  stage5_full_page)
    export DOCGRID_PARENT_CHECKPOINT="${DOCGRID_PARENT_CHECKPOINT:-${DOCGRID_RUN_ROOT}/stage4_qwen/seed-${DOCGRID_SEED}/best.pt}"
    export DOCGRID_OUTPUT_DIR="${DOCGRID_OUTPUT_DIR:-${DOCGRID_RUN_ROOT}/stage5_full_page/seed-${DOCGRID_SEED}}"
    export DOCGRID_STAGE5_TRAIN_MANIFEST="${DOCGRID_STAGE5_TRAIN_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/train.jsonl}"
    export DOCGRID_STAGE5_VAL_MANIFEST="${DOCGRID_STAGE5_VAL_MANIFEST:-${DOCGRID_DATA_ROOT}/full_page/manifests/val.jsonl}"
    export DOCGRID_STAGE5_FROZEN_CONTRACT="${DOCGRID_STAGE5_FROZEN_CONTRACT:-${DOCGRID_RUN_ROOT}/stage0_full_page_audit/frozen_contract.json}"
    ;;
  *)
    echo "unknown DocGrid stage: ${stage}" >&2
    exit 2
    ;;
esac

docgrid_run_container "${DOCGRID_TRAIN_CPUS:-8}" \
  'bash cp_docflow_v1/slurm/docgrid_v2/train_stage.sbatch'
