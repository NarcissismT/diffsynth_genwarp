#!/usr/bin/env bash
# Portal resources: 1x A800, 8 CPU, 192G host RAM, at least 8 hours.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_container.sh"

export DOCGRID_QWEN_MODEL="${DOCGRID_QWEN_MODEL:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit}"
export DOCGRID_QWEN_PROBE_IMAGE="${DOCGRID_QWEN_PROBE_IMAGE:-/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/img/Doc3d_crop/Doc3d_crop_0000000.png}"
export DOCGRID_QWEN_VALIDATION_REPORT="${DOCGRID_QWEN_VALIDATION_REPORT:-${DOCGRID_RUN_ROOT}/qwen_runtime_validation/report.json}"
docgrid_run_container 8 \
  'bash cp_docflow_v1/slurm/docgrid_v2/validate_qwen_runtime.sbatch'
