#!/usr/bin/env bash
# =============================================================================
# V4.4 + 内嵌验证集（按推理那种：验证 main 图用预生成的真扩散 corrected_low）
#
# 与 train_v4_4_layer36.sh 的唯一区别：
#   1. csv 换成 metadata_train.csv（已剔除 held-out 验证样本）
#   2. 传 --val_index_csv 开启训练内嵌验证，每 --val_interval 优化步报一次
#      验证 EPE/<5px（分 low/mid/high），追加到 output_path/val_metrics.csv
#
# 前置：先跑 scripts/prepare_val_set.sh 产出 train/val split + 预生成 corrected_low。
# =============================================================================

# ★ 带验证集版本：训练用 metadata_train.csv（已剔除验证样本），验证用预生成 corrected_low
csv_path=/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_train.csv
val_index_csv=/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/val_corrected_low/val_index.csv
train_id=20260610-v4_4-vaeroundtrip-layer36-val

# 纯 DiT LoRA（用于特征提取，冻结不训练）
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

# 切到本脚本所在版本目录运行（import 本地 utils）
cd "$(dirname "$0")/.."

accelerate launch train_flow_head_v4_4.py \
  --dataset_base_path $(dirname "$csv_path") \
  --dataset_metadata_path $csv_path \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 5 \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "flow_head." \
  --output_path "/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_4_ckpts/$train_id" \
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
  --dit_target_layers "35" \
  --diff_out_ch 96 \
  --k_source "warped" \
  --hard_weight_alpha 1.0 \
  --hard_weight_ref 24.0 \
  --hard_weight_max 3.0 \
  --gradient_accumulation_steps 2 \
  --vae_roundtrip_prob 0.5 \
  --loss_print_interval 100 \
  --val_index_csv "$val_index_csv" \
  --val_interval 4000 \
  --model_paths "$MODEL_PATHS"
