#!/bin/bash

train_id=20250904-2
ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/$train_id/qwen-image-edit"
test_set="53"
crop=True
if crop=True; then
  crop_postfix=""
else
  crop_postfix="-no_crop"
fi

# 运行 Python 脚本，传递 --crop 参数
CUDA_VISIBLE_DEVICES=4 python qwen_image_batch_full_lora.py \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test/$test_set" \
  --output_dir "$ckpt.visualization/$test_set$crop_postfix" \
  --prompt "将这张图片变清晰、去除模糊" \
  --batch_size 1 \
  --crop $crop