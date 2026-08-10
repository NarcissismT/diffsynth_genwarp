#!/bin/bash



ckpt="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250912-2/step-63500.safetensors"
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
  --output_dir "$ckpt.visualization/0916-5_bad2_steps10_0912-2-63500_1024_$resize_mode" \
  --prompt "Flatten this warped or curled document image by correcting any distortions, while preserving the original layout, text clarity, and visual details." \
  --batch_size 1 \
  --short_side_pixel 1024 \
  --img_size 1024 \
  --infer_steps 10 \
  $resize_mode_arg  # 这里传递 crop 参数
