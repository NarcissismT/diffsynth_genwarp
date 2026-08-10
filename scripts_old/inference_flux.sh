#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python flux-kontext-batch.py \
  --lora_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/lora/FLUX.1-Kontext-dev_lora/20250825/dehighlight_compareyuyi/step-53475.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/junle_liu/滤镜/证件测试集采样/certificate/reflection_sample/" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250825/flux_kontext_lora/step-53475" \
  --prompt "Enhance this image to high definition, remove reflections, and balance light and shadow." \
  --batch_size 1