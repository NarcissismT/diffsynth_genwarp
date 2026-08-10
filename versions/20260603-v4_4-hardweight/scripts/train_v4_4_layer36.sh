#!/usr/bin/env bash
# =============================================================================
# V4.4 (path B + 难度加权 + 梯度累积)：单层 Layer 36，双图拼接，Q←corrected / K←warped（跨图）
#
# 修复 V4.1 的根因：V4.1 的 Q/K 来自同一张图 → 无跨图位移信号 → 水波纹。
# V4.2 利用 Qwen-Image-Edit 原生跨图 attention：把 corrected(main,加噪) 和
# warped(edit,干净) 拼进同一序列，Q 切 corrected 段、K 切 warped 段。
#
# 与 sham 对照 (train_v4_2_sham_layer36.sh, k_source=corrected) 配对跑，
# 用 K-source 消融判定"原生跨图注意力是否携带几何信号"。
# =============================================================================

# 用带 corrected_vae_path 列的 CSV（VAE round-trip 域对齐）
csv_path=/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_vae.csv
train_id=20260610-v4_4-vaeroundtrip-layer36

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
  --model_paths "$MODEL_PATHS"
