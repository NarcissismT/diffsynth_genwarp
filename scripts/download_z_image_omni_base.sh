#!/usr/bin/env bash
set -Eeuo pipefail

container_image="${CONTAINER_IMAGE:-registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
model_dir="${Z_IMAGE_MODEL_DIR:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Z-Image-Omni-Base}"
hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
hf_home="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"

docker_args=(
  run --rm
  --user "$(id -u):$(id -g)"
  --entrypoint hf
  -e HOME=/tmp
  -e "HF_ENDPOINT=$hf_endpoint"
  -e "HF_HOME=$hf_home"
  -v /juicefs-algorithm:/juicefs-algorithm
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(-e HF_TOKEN)
fi

printf '%s\n' \
  "Downloading the gated Z-Image-Omni-Base transformer and SigLIP weights to:" \
  "  $model_dir"

docker "${docker_args[@]}" "$container_image" download \
  Tongyi-MAI/Z-Image-Omni-Base \
  --include 'transformer/*' 'siglip/*' \
  --local-dir "$model_dir"
