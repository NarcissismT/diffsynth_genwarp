#!/bin/bash
# =============================================================================
# Flow Direct 推理脚本（无 RAFT，直接用 FlowHead 输出光流）
# 对应脚本: qwen_image_flow_direct.py
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2/表格线_101992.jpg"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_direct_results"
# 联合训练 checkpoint（包含 LoRA + FlowHead）
CHECKPOINT="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_ckpts/20260509-1_1in10_w_flow_head/step-212000.safetensors"

# ---- 推理参数 ----
IMG_SIZE=1024
INFER_STEPS=50
RESIZE_MODE="stretch"

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

$PYTHON qwen_image_flow_direct.py \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --checkpoint  "$CHECKPOINT" \
    --prompt      "$PROMPT" \
    --img_size    $IMG_SIZE \
    --infer_steps $INFER_STEPS \
    --resize_mode $RESIZE_MODE \
    --batch_size  1
