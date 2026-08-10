#!/bin/bash
# =============================================================================
# V4.2 high-only 长训过拟合诊断（单卡，radius=4 架构）
# 判断 <5px 卡 40-50% 是【容量瓶颈】还是【数据/训练问题】
#
# 在 18 个固定分层样本上反复训练 3000 步，看各层能否过拟合到 ~100% <5px。
# 单卡即可（过拟合诊断不需要 DDP）。约 1 个下午（含模型加载）。
# =============================================================================

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

LORA_CKPT="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# Python 解释器（容器内用系统 python）
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

cd "$(dirname "$0")/.."

# 单卡，限制只用 1 张 GPU
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON overfit_high_only.py \
    --model_paths "$MODEL_PATHS" \
    --tokenizer_path "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/tokenizer" \
    --processor_path "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/processor" \
    --lora_checkpoint "$LORA_CKPT" \
    --dit_target_layers "35" \
    --k_source "warped" \
    --diff_out_ch 96 \
    --max_pixels 1048576 \
    --lambda_flow 1.0 \
    --lambda_warp 0.5 \
    --gradloss_ratio 1.0 \
    --learning_rate 3e-4 \
    --num_steps 8000 \
    --eval_every 200 \
    --train_iters 4 \
    --eval_iters 4 \
    --only_layer "high"
