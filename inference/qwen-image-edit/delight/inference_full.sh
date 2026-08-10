#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python qwen_image_batch_full.py \
  --dit_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/qwen-image-edit/20250822/dehighlight_compareyuyi/step-2000.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/junle_liu/滤镜/证件测试集采样/certificate/reflection_sample/驾驶证" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250822/full/dehighlight_compareyuyi_lora/step-2000" \
  --prompt "将这张图片变高清，去除反光，均衡光影" \
  --batch_size 1