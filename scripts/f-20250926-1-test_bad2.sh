#!/bin/bash



ckpt="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250919-1_1in10_z_unwarp/step-332000.safetensors"
# test_set="SDXL0829_测试集效果不佳"
test_set="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad_tmp"
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
  --output_dir "$ckpt.visualization/0926-1_bad2_1024_$resize_mode" \
  --prompt "Apply geometric correction to the image to eliminate skew, curl, or distortion, producing a visually flattened, front-facing result where textual content and structural elements such as table lines appear horizontally or vertically aligned. Preserve the original content and stylistic characteristics without altering or regenerating any visual elements. If geometric adjustment results in undefined or empty regions, fill those areas with solid black pixels. Retain all foreground elements, including possible finger occlusions, as part of the original image. This is a pixel-level transformation task, not a content generation task." \
  --batch_size 1 \
  --short_side_pixel 1024 \
  --img_size 1024 \
  --infer_steps 50 \
  $resize_mode_arg  # 这里传递 crop 参数
