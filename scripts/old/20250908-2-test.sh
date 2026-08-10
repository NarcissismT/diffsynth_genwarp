#!/bin/bash

ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/20250908-2/step-36700.safetensors"
# test_set="SDXL0829_测试集效果不佳"
test_set="train/bm1"
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
CUDA_VISIBLE_DEVICES=0 python qwen_image_batch_full_lora.py \
  --lora_path $ckpt \
  --input_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test/$test_set" \
  --output_dir "$ckpt.visualization/$test_set$crop_postfix" \
  --prompt "输入的图像是一张扭曲的文本图像，请将这张图像矫正为一张平整的图片。请输出一张图片，代表两个方向上的位移场，平整图片每个像素施加对应位置的位移后就是原扭曲图片。位移场图片的第一个通道是水平方向，第二个通道是垂直方向，第三个通道为 0。" \
  --batch_size 1 \
  $crop_arg  # 这里传递 crop 参数
