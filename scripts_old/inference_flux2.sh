#!/bin/bash

for i in $(seq 21 40)
do
CUDA_VISIBLE_DEVICES=2 python flux-kontext-batch.py \
  --lora_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/lora/FLUX.1-Kontext-dev_lora/20250825/dehighlight_compareyuyi/step-53475.safetensors" \
  --input_dir "/juicefs-algorithm/data/IPT/junyan_cao/datasets/cs_classificated/certificate/$i" \
  --output_dir "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/20250825/flux_kontext_lora/step-53475-certificate/$i" \
  --prompt "Enhance this image to high definition, remove reflections, and balance light and shadow." \
  --batch_size 1
done