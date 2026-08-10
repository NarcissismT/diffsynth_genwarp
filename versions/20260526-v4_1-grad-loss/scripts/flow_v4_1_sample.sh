#!/bin/bash
# =============================================================================
# Flow V4.1 推理脚本（V4 + gradient loss + InstanceNorm 训练得到）
# 对应训练：scripts/train_v4_1_exp_C_layer36.sh
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v4_1_results"

# ---- FlowHead V4.1 checkpoint ----
# 实验 C：单层 Layer 36（首选）
CKPT_C="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_1_ckpts/20260526-v4_1-exp_C_layer36/step-435060.safetensors"
DIT_LAYERS_C="35"

# 实验 E：三层融合
CKPT_E="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_1_ckpts/20260526-v4_1-exp_E_layer24_36_48/step-435060.safetensors"
DIT_LAYERS_E="23,35,47"

# 选择实验
CKPT="$CKPT_C"
DIT_LAYERS="$DIT_LAYERS_C"
EXP_SUFFIX="exp_C_layer36"

# DiT LoRA
LORA_PATH="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# ---- 推理参数 ----
IMG_SIZE=512        # 与训练分辨率匹配
INFER_STEPS=50
RESIZE_MODE="stretch"
FLOW_ITERS=4        # 与训练 iters=4 对齐

# ---- Prompt ----
PROMPT="Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately."

# ---- Python 解释器 ----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

# 切到本脚本所在版本目录运行（因为脚本里用了相对 import utils）
cd "$(dirname "$0")/.."

$PYTHON qwen_image_flow_v4_1.py \
    --input_dir         "$INPUT_DIR" \
    --output_dir        "${OUTPUT_DIR}/${EXP_SUFFIX}" \
    --ckpt_path         "$CKPT" \
    --dit_target_layers "$DIT_LAYERS" \
    --lora_path         "$LORA_PATH" \
    --img_size          $IMG_SIZE \
    --infer_steps       $INFER_STEPS \
    --resize_mode       $RESIZE_MODE \
    --flow_iters        $FLOW_ITERS \
    --save_flow_vis \
    --prompt            "$PROMPT"
