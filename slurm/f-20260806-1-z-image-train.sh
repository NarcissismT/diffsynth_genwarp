#!/bin/bash
# default bash
set -Eeuo pipefail

export HF_HOME=/juicefs-algorithm/data/IPT/yuang_feng/cache
export TRITON_CACHE_DIR=/tmp/slurm_${SLURM_JOB_ID}/triton
export TORCH_EXTENSIONS_DIR=/tmp/slurm_${SLURM_JOB_ID}/deepspeed_cache

srun --cpus-per-task 128 -K \
  --container-image=docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers \
  --container-mounts=/juicefs-algorithm:/juicefs-algorithm \
  --container-workdir=/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp \
  --container-env=HF_HOME,TRITON_CACHE_DIR,TORCH_EXTENSIONS_DIR \
  bash scripts/f-20260806-1-z-image-train.sh
