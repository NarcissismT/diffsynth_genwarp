#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python flux-kontext-batch_full.py \
  --dit_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/FLUX.1-Kontext-dev_lora/20250826/dehighlight_compareyuyi/step-3000.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/junle_liu/滤镜/证件测试集采样/certificate/reflection_sample/" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250826/flux_kontext_full/step-3000" \
  --prompt "Enhance this image to high definition, remove reflections, and balance light and shadow." \
  --batch_size 1