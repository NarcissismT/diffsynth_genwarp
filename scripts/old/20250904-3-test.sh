#!/bin/bash

train_id=20250904-3
ckpt="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/$train_id/qwen-image-edit"
test_root="/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/test"
# test_set="SDXL0829_测试集效果不佳"
test_set="53"
# test_set="temp"
crop=True
if crop=True; then
  crop_postfix=""
else
  crop_postfix="-no_crop"
fi

CUDA_VISIBLE_DEVICES=4 python qwen_image_batch_full_lora.py \
  --input_dir "$test_root/$test_set" \
  --output_dir "$ckpt.visualization/$test_set$crop_postfix" \
  --prompt_dict "$test_root/$test_set/metadata_ocr.csv" \
  --batch_size 1 \
  --crop $crop