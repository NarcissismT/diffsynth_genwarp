#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/20250903-1/step-5460.safetensors"

total_divide=8  # 总共分成多少份
avaliable_gpu_indexes=(0 1 2 4 5 6 7 8)
test_set="SDXL0829_测试集效果不佳"
# test_set="cs_classified"
# test_set="temp"
# test_set="53"
crop=true  # 设置 crop 为 true 或 false
# enable_dit_fp8_computation=true  # 是否启用DIT的FP8计算以节省显存
enable_dit_fp8_computation=false  # 是否启用DIT的FP8计算以节省显存

short_side_pixel=3072
img_size=2048

# 如果 crop 为 true，传递 --crop 参数
if [ "$crop" == true ]; then
  crop_postfix=""  # 启用裁切
  crop_arg="--crop"
else
  crop_postfix="-no_crop"
  crop_arg=""
fi

if [ "$enable_dit_fp8_computation" == true ]; then
  enable_dit_fp8_computation_postfix="-fp8"
  enable_dit_fp8_computation_arg="--enable_dit_fp8_computation"
else
  enable_dit_fp8_computation_postfix=""
  enable_dit_fp8_computation_arg=""
fi

short_side_pixel_postfix="-$short_side_pixel"
img_size_postfix="-img_size_$img_size"

for ((divide_index=0; divide_index<total_divide; divide_index++)); do
  CUDA_VISIBLE_DEVICES=${avaliable_gpu_indexes[$divide_index]} python qwen_image_batch_full_lora.py \
    --lora_path $ckpt \
    --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test/$test_set" \
    --output_dir "$ckpt.visualization/$test_set$crop_postfix$short_side_pixel_postfix$img_size_postfix$enable_dit_fp8_computation_postfix" \
    --prompt "将这张图片变清晰、去除模糊" \
    --batch_size 1 \
    --short_side_pixel $short_side_pixel \
    --img_size $img_size \
    --total_divide $total_divide \
    --divide_index $divide_index \
    $crop_arg \
    $enable_dit_fp8_computation_arg &
done

# 等待所有子进程完成
wait
echo "所有任务完成 ✅"
