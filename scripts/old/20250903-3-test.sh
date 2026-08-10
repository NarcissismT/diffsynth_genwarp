#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/20250903-3/step-5460.safetensors"
test_root="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test"
# test_set="SDXL0829_测试集效果不佳"
test_set="53"
# test_set="temp"
crop=True

CUDA_VISIBLE_DEVICES=4 python qwen_image_batch_full_lora.py \
  --lora_path $ckpt \
  --input_dir "$test_root/$test_set" \
  --output_dir "$ckpt.visualization/$test_set" \
  --prompt_dict "$test_root/$test_set/metadata_ocr.csv" \
  --batch_size 1
  --crop $crop