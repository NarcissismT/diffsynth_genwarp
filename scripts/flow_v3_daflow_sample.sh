#!/bin/bash
# =============================================================================
# Flow V3 DA-Flow 推理脚本
# 对应脚本: qwen_image_flow_v3_daflow.py
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2/表格线_101790.jpg"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v3_daflow_results"
# 联合训练 checkpoint（LoRA + FlowHead V3）或纯 LoRA checkpoint
LORA_PATH="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v3_daflow_ckpts/20260513-1_1in10_w_flow_head_v3_daflow/step-104000.safetensors"
# 如果 checkpoint 不含 FlowHead V3，单独指定（可留空）
FLOW_HEAD_V3_PATH=""

# ---- 推理参数 ----
IMG_SIZE=1024
INFER_STEPS=50
RESIZE_MODE="stretch"
FLOW_ITERS=12       # FlowHead V3 迭代次数
NUM_DIT_LAYERS=4    # 提取 DiT top-L 层特征（与训练时一致）

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

$PYTHON qwen_image_flow_v3_daflow.py \
    --input_dir         "$INPUT_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --lora_path         "$LORA_PATH" \
    --flow_head_v3_path "$FLOW_HEAD_V3_PATH" \
    --prompt            "$PROMPT" \
    --img_size          $IMG_SIZE \
    --infer_steps       $INFER_STEPS \
    --resize_mode       $RESIZE_MODE \
    --flow_iters        $FLOW_ITERS \
    --num_dit_layers    $NUM_DIT_LAYERS \
    --batch_size        1
