#!/bin/bash
# 启动 diffsynth-genwarp 容器（交互式）
# 用法：bash docker/run.sh [tag] [额外 docker run 参数]
# 默认 tag: diffsynth-genwarp:latest

TAG="${1:-diffsynth-genwarp:latest}"
shift 2>/dev/null

docker run --gpus all --rm -it \
  --shm-size=64G \
  -v /juicefs-algorithm:/juicefs-algorithm:rw \
  -v /home/zhuochu_yang:/home/zhuochu_yang \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --user "$(id -u):$(id -g)" \
  -w /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp \
  -e PYTHONNOUSERSITE=1 \
  "$@" \
  "$TAG" \
  bash
