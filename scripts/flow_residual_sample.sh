#!/bin/bash
# =============================================================================
# Flow Residual 推理示例脚本
# 对应脚本: qwen_image_flow_residual.py
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2/表格线_101790.jpg"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_residual_results"
# 联合训练 checkpoint（同时包含 LoRA + FlowHead）
LORA_PATH="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_ckpts/20260509-1_1in10_w_flow_head/step-144000.safetensors"
FLOW_HEAD_PATH=""   # 联合训练 checkpoint 已内嵌 FlowHead，留空即可

# ---- 推理参数 ----
IMG_SIZE=1024
INFER_STEPS=50
RESIZE_MODE="stretch"

# ---- 光流参数 ----
RAFT_SIZE="large"
FLOW_ITERS=20

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

$PYTHON qwen_image_flow_residual.py \
    --input_dir        "$INPUT_DIR" \
    --output_dir       "$OUTPUT_DIR" \
    --lora_path        "$LORA_PATH" \
    --flow_head_path   "$FLOW_HEAD_PATH" \
    --prompt           "$PROMPT" \
    --img_size         $IMG_SIZE \
    --infer_steps      $INFER_STEPS \
    --resize_mode      $RESIZE_MODE \
    --raft_model_size  $RAFT_SIZE \
    --num_flow_updates $FLOW_ITERS \
    --batch_size       1
