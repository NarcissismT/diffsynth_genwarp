#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/20250904-10/step-10000.safetensors"
# test_set="SDXL0829_测试集效果不佳"
test_set="train/quark"
crop=false  # 设置 crop 为 true 或 false

# 如果 crop 为 true，传递 --crop 参数
if [ "$crop" == true ]; then
  crop_postfix=""  # 启用裁切
  crop_arg="--crop"  # 传递 --crop
else
  crop_postfix="-no_crop"  # 不裁切
  crop_arg=""  # 不传递 --crop
fi


# 运行 Python 脚本，传递 --crop 参数
CUDA_VISIBLE_DEVICES=7 python qwen_image_batch_full_lora.py \
  --lora_path $ckpt \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test/$test_set" \
  --output_dir "$ckpt.visualization/$test_set$crop_postfix" \
  --prompt "将这张图片变清晰、去除模糊" \
  --batch_size 1 \
  $crop_arg  # 这里传递 crop 参数
