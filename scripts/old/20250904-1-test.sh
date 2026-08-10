#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/20250904-1/step-5460.safetensors"
test_set="53"
crop=True
if crop=True; then
  crop_postfix=""
else
  crop_postfix="-no_crop"
fi

# 运行 Python 脚本，传递 --crop 参数
CUDA_VISIBLE_DEVICES=5 python qwen_image_batch_full_lora.py \
  --lora_path $ckpt \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test/$test_set" \
  --output_dir "$ckpt.visualization/$test_set$crop_postfix" \
  --prompt "你识别图片中的所有文字内容，然后将这个图片变清晰" \
  --batch_size 1 \
  --crop $crop
