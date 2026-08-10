#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python qwen_image_batch.py \
  --lora_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/models/train/Qwen-Image-Edit_lora/20250821/zhongdu_mohutuihua/step-100.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/test" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results" \
  --prompt "对这张清晰图像应用中等程度的模糊效果，模拟图像退化过程。" \
  --batch_size 1