#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/20250903-2/step-10000.safetensors"
# test_set="SDXL0829_测试集效果不佳"
# test_set="cs_classified"
# test_set="temp"
test_set="53"
crop=true  # 设置 crop 为 true 或 false
# enable_dit_fp8_computation=true  # 是否启用DIT的FP8计算以节省显存
enable_dit_fp8_computation=false  # 是否启用DIT的FP8计算以节省显存

# short_side_pixel=4096
# img_size=2048
# ORI: 66446 / 81920 MB 15:38<00:00

# short_side_pixel=2536
# img_size=2048
# ORI: 62036 / 81920 MB 06:28<00:00
# FP8: 43706 / 81920 MB 06:28<00:00

# short_side_pixel=2048
# img_size=1024 
# ORI: 58718 / 81920 MB 02:14<00:00
# FP8: 41954 / 81920 MB

# short_side_pixel=3072
# img_size=2048
# ORI: 62036 / 81920 MB 06:28<00:00
# FP8: 43706 / 81920 MB 06:28<00:00

short_side_pixel=3072
img_size=1536 







# 如果 crop 为 true，传递 --crop 参数
if [ "$crop" == true ]; then
  crop_postfix=""  # 启用裁切
  crop_arg="--crop"  # 传递 --crop
else
  crop_postfix="-no_crop"  # 不裁切
  crop_arg=""  # 不传递 --crop
fi

if [ "$enable_dit_fp8_computation" == true ]; then
  enable_dit_fp8_computation_postfix="-fp8"  # 启用FP8计算
  enable_dit_fp8_computation_arg="--enable_dit_fp8_computation"  # 传递 --enable_dit_fp8_computation
else
  enable_dit_fp8_computation_postfix=""  # 不启用FP8计算
  enable_dit_fp8_computation_arg=""  # 不传递 --enable_dit_fp8_computation
fi

short_side_pixel_postfix="-$short_side_pixel"
img_size_postfix="-img_size_$img_size"

# 运行 Python 脚本，传递 --crop 参数
CUDA_VISIBLE_DEVICES=4 python qwen_image_batch_full_lora.py \
  --lora_path $ckpt \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test/$test_set" \
  --output_dir "$ckpt.visualization/$test_set$crop_postfix$short_side_pixel_postfix$img_size_postfix$enable_dit_fp8_computation_postfix" \
  --prompt "将这张图片变清晰、去除模糊" \
  --batch_size 1 \
  --short_side_pixel $short_side_pixel \
  --img_size $img_size \
  $crop_arg \
  $enable_dit_fp8_computation_arg
