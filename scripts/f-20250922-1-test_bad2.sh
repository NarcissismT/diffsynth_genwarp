#!/bin/bash



ckpt="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250919-1_1in10_z_unwarp/step-112000.safetensors"
# test_set="SDXL0829_测试集效果不佳"
test_set="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2"
# test_set="train/quark/img"

resize_mode="stretch"  # 设置 crop 为 true 或 false
# resize_mode="scale_to_short_side"  

resize_mode_arg="--resize_mode=$resize_mode"

# # 如果 crop 为 true，传递 --crop 参数
# if [ "$crop" == true ]; then
#   crop_postfix=""  # 启用裁切
#   crop_arg="--crop"  # 传递 --crop
# else
#   crop_postfix="-no_crop"  # 不裁切
#   crop_arg=""  # 不传递 --crop
# fi

  # --input_dir "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/data/$test_set" \

# 运行 Python 脚本，传递 --crop 参数
 python /juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/qwen_image_batch_full_lora.py \
  --lora_path $ckpt \
  --input_dir "$test_set" \
  --output_dir "$ckpt.visualization/0923-1_bad2_0919-1-112000_1024_$resize_mode" \
  --prompt "Given an input image of a document that may be skewed, curled, or distorted, generate an output image that is geometrically corrected to appear flat and front-facing. Ensure that all original content and text clarity are fully preserved. This is a correction task—do not regenerate or alter any content beyond geometric restoration." \
  --batch_size 1 \
  --short_side_pixel 1024 \
  --img_size 1024 \
  --infer_steps 50 \
  $resize_mode_arg  # 这里传递 crop 参数
