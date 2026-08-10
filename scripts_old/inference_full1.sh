#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python qwen_image_batch_full_RealPDF1.py \
  --dit_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/qwen-image-edit/20280828/full/RealPDF/step-2300.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250825/flux_kontext_lora/step-2000" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250826/full/RealPDF/step-2300-fanguang" \
  --prompt "将这张图片变清晰、去除模糊" \
  --batch_size 1