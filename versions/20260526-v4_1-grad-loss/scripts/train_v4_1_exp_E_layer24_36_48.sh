#!/usr/bin/env bash
# V4.1 实验 E：三层融合 Layer 24+36+48 + gradient_loss + InstanceNorm
# 从头训练

csv_path=/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_flow.csv
train_id=20260526-v4_1-exp_E_layer24_36_48

LORA_CKPT="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

MODEL_PATHS='[
    [
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
    ],
    [
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00001-of-00004.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00002-of-00004.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00003-of-00004.safetensors",
        "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00004-of-00004.safetensors"
    ],
    "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors"
]'

cd "$(dirname "$0")/.."

accelerate launch train_flow_head_v4_1_grad.py \
  --dataset_base_path $(dirname "$csv_path") \
  --dataset_metadata_path $csv_path \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 5 \
  --learning_rate 1e-4 \
  --num_epochs 6 \
  --remove_prefix_in_ckpt "flow_head." \
  --output_path "/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_1_ckpts/$train_id" \
  --use_gradient_checkpointing \
  --dataset_num_workers 2 \
  --find_unused_parameters \
  --save_steps 4000 \
  --tokenizer_path "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/tokenizer" \
  --processor_path "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/processor" \
  --lora_checkpoint "$LORA_CKPT" \
  --lambda_flow 1.0 \
  --lambda_warp 0.5 \
  --gradloss_ratio 1.0 \
  --dit_target_layers "23,35,47" \
  --diff_out_ch 96 \
  --model_paths "$MODEL_PATHS"
