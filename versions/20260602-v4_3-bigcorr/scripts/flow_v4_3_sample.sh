#!/bin/bash
# =============================================================================
# Flow V4.2 Path B 推理脚本
# 对应训练：scripts/train_v4_2_pathB_layer36.sh（k_source=warped, layer=35）
#
# 推理流程：
#   warped → Qwen 50 步扩散 → corrected_low
#   corrected_low + warped 拼接喂 DiT → 跨段提取 Q/K（Q←corrected_low, K←warped）
#   FlowHeadV4.2 → flow → 上采样 → warp 原始高清图
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v4_3_results"

# ---- FlowHead V4.2 checkpoint（path B，单层 Layer 36）----
CKPT="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_3_ckpts/20260602-v4_3-bigcorr-layer36/step-REPLACE.safetensors"
DIT_LAYERS="35"
K_SOURCE="warped"        # 必须与训练一致（pathB=warped）
EXP_SUFFIX="v4_3_layer36"

# ---- DiT LoRA（与训练一致）----
LORA_PATH="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# ---- 推理参数 ----
IMG_SIZE=512        # 必须与训练分辨率一致
INFER_STEPS=50
RESIZE_MODE="stretch"
FLOW_ITERS=12        # 与训练 iters=4 对齐

# ---- Prompt ----
PROMPT="Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately."

# ---- Python 解释器 ----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

# 切到本脚本所在版本目录运行（import 本地 utils）
cd "$(dirname "$0")/.."

$PYTHON qwen_image_flow_v4_3.py \
    --input_dir         "$INPUT_DIR" \
    --output_dir        "${OUTPUT_DIR}/${EXP_SUFFIX}" \
    --ckpt_path         "$CKPT" \
    --dit_target_layers "$DIT_LAYERS" \
    --k_source          "$K_SOURCE" \
    --lora_path         "$LORA_PATH" \
    --img_size          $IMG_SIZE \
    --infer_steps       $INFER_STEPS \
    --resize_mode       $RESIZE_MODE \
    --flow_iters        $FLOW_ITERS \
    --save_flow_vis \
    --prompt            "$PROMPT"
