#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-200000.safetensors"
test_set_root="/juicefs-algorithm/data/IPT/yuang_feng/DATA/0915_realall_split8"

resize_mode="stretch"  # 设置 crop 为 true 或 false
# resize_mode="scale_to_short_side"  

resize_mode_arg="--resize_mode=$resize_mode"
# crop=false  # 设置 crop 为 true 或 false

# # 如果 crop 为 true，传递 --crop 参数
# if [ "$crop" == true ]; then
#   crop_postfix=""  # 启用裁切
#   crop_arg="--crop"
# else
#   crop_postfix="-no_crop"
#   crop_arg=""
# fi

# 设置 GPU 索引（根据你机器的 GPU 数量调整）
avaliable_gpu_indexes=(0 1 2 3 4 5 6 7)

# 启动 8 个进程，每个处理一个分片
for ((i=0; i<8; i++)); do
  input_dir="$test_set_root/split_$i"
  output_dir="$ckpt.visualization/1013-2_realall-sp8_0929-1-200000_1024_$resize_mode/split_$i"

  CUDA_VISIBLE_DEVICES=${avaliable_gpu_indexes[$i]} python /juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/qwen_image_batch_full_lora.py \
    --lora_path "$ckpt" \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --prompt "Apply geometric correction to the input image to eliminate distortions such as skew, curl, folds, or non-frontal perspective, producing a flat, front-facing image where all textual content and structural elements, including table lines, are strictly aligned horizontally or vertically, resembling the layout of a standard PDF document. Preserve all original document content without alteration or regeneration. Retain all foreground elements, including any finger occlusions or shadows. If geometric correction creates undefined or empty regions at the edges, fill those areas with solid white pixels." \
    --batch_size 1 \
    --short_side_pixel 1024 \
    --infer_steps 50 \
    $resize_mode_arg &
done

# 等待所有子进程完成
wait
echo "所有分片任务完成 ✅"
