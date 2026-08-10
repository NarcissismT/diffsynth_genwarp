#!/bin/bash
# 构建 diffsynth-genwarp 镜像
# 用法：bash docker/build.sh [tag]
# 默认 tag: diffsynth-genwarp:latest

TAG="${1:-diffsynth-genwarp:latest}"
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t "$TAG" .
echo "构建完成: $TAG"
