#!/bin/bash
# =============================================================================
# Flow V2 推理脚本（DiT LoRA + FlowHead V2，无 RAFT）
# 对应脚本: qwen_image_flow_v2.py
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2/表格线_101790.jpg"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v2_results"
# DiT LoRA 权重（使用最新 checkpoint）
LORA_PATH="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_ckpts/20260512-1_1in10_w_flow_head_v2/step-172000.safetensors"
# FlowHead V2 权重（训练完成后填写）
FLOW_HEAD_V2_PATH=""

# ---- 推理参数 ----
IMG_SIZE=1024
INFER_STEPS=50
RESIZE_MODE="stretch"
FLOW_ITERS=12   # FlowHead V2 推理迭代次数

# ---- Prompt ----
PROMPT="Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately."

# ---- Python 解释器 ----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

# ---- 运行 ----
cd "$(dirname "$0")/.."

$PYTHON qwen_image_flow_v2.py \
    --input_dir         "$INPUT_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --lora_path         "$LORA_PATH" \
    --flow_head_v2_path "$FLOW_HEAD_V2_PATH" \
    --prompt            "$PROMPT" \
    --img_size          $IMG_SIZE \
    --infer_steps       $INFER_STEPS \
    --resize_mode       $RESIZE_MODE \
    --flow_iters        $FLOW_ITERS \
    --batch_size        1
