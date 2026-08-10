#!/bin/bash

for i in $(seq 21 40)
do
  input_dir="/juicefs-algorithm/data/IPT/junyan_cao/datasets/cs_classificated/certificate/$i"
  output_dir="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250826/full/RealPDF/step-15200-1-certificate/$i"
  echo "input_dir: $input_dir"
  echo "output_dir: $output_dir"
  CUDA_VISIBLE_DEVICES=4 python qwen_image_batch_full_RealPDF.py \
  --dit_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/qwen-image-edit/20280828/full/RealPDF/step-15200.safetensors" \
  --input_dir "$input_dir" \
  --output_dir "$output_dir" \
  --prompt "将这张图片变清晰、去除模糊" \
  --batch_size 1
done
