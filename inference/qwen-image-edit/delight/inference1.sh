#!/bin/bash

CUDA_VISIBLE_DEVICES=8 python qwen_image_batch.py \
  --lora_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/models/train/qwen-image-edit/20250822/lora/dehighlight_compareyuyi_lora/step-5000.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/junle_liu/滤镜/证件测试集采样/certificate/reflection_sample/驾驶证" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250822/lora/dehighlight_compareyuyi_lora/step-5000" \
  --prompt "将这张图片变高清，去除反光，均衡光影" \
  --batch_size 1