#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250912-2/step-63500.safetensors"
test_set_root="/juicefs-algorithm/data/IPT/yuang_feng/DATA/20250915_realData_split8"
crop=false  # 设置 crop 为 true 或 false

# 如果 crop 为 true，传递 --crop 参数
if [ "$crop" == true ]; then
  crop_postfix=""  # 启用裁切
  crop_arg="--crop"
else
  crop_postfix="-no_crop"
  crop_arg=""
fi

# 设置 GPU 索引（根据你机器的 GPU 数量调整）
avaliable_gpu_indexes=(0 1 2 3 4 5 6 7)

# 启动 8 个进程，每个处理一个分片
for ((i=0; i<8; i++)); do
  input_dir="$test_set_root/split_$i"
  output_dir="$ckpt.visualization/0915-3_realdata-sp8_0912-2-63500_1024$crop_postfix/split_$i"

  CUDA_VISIBLE_DEVICES=${avaliable_gpu_indexes[$i]} python /juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/qwen_image_batch_full_lora.py \
    --lora_path "$ckpt" \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --prompt "Flatten this warped or curled document image by correcting any distortions, while preserving the original layout, text clarity, and visual details." \
    --batch_size 1 \
    --short_side_pixel 1024 \
    --img_size 1024 \
    $crop_arg &
done

# 等待所有子进程完成
wait
echo "所有分片任务完成 ✅"
