#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python qwen_image_batch_full_RealPDF.py \
  --dit_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/qwen-image-edit/20280828/full/RealPDF/step-3000.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/模糊+粘连" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250826/full/RealPDF/step-3000" \
  --prompt "将这张图片变清晰、去除模糊" \
  --batch_size 1