#!/bin/bash
# =============================================================================
# Flow V4 推理脚本（Layer Probing 版）
# 对应脚本: qwen_image_flow_v4.py
#
# 正确推理流程（与 V3 一致）：
#   Stage 1: Qwen 扩散完整推理 → corrected_low + Q/K 特征（指定 target_layers）
#   Stage 2: FlowHeadV4(corrected_low, warped, q_feats, k_feats) → flow
#   Stage 3: 上采样 flow 到原始高清分辨率
#   Stage 4: warp 原始高清图 → 最终输出（文字来自原图像素）
#
# 输出：
#   a_*.jpg  推理输入（resize 后的 warped 图）
#   b_*.jpg  扩散结果（corrected_low，仅供参考）
#   c_*.jpg  FlowHead V4 warp 高清结果（最终输出）
#   d_*.jpg  三合一对比图（原始 | 扩散 | V4 warp）
# =============================================================================

# ---- 路径配置 ----
INPUT_DIR="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v4_results_fixed"

# ---- FlowHead V4 checkpoint（选择实验 C 或 E 的最新 ckpt）----
# 实验 C：单层 Layer 36（推荐首选）
CKPT_C="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/20260515-exp_C_layer36/step-240000.safetensors"
DIT_LAYERS_C="35"

# 实验 E：三层融合 Layer 24+36+48
CKPT_E="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/20260515-exp_E_layer24_36_48/step-228000.safetensors"
DIT_LAYERS_E="23,35,47"

# ---- 选择使用哪个实验（修改下面两行）----
CKPT="$CKPT_C"
DIT_LAYERS="$DIT_LAYERS_C"
EXP_SUFFIX="exp_C_layer36"

# ---- DiT LoRA（用于提升特征质量）----
LORA_PATH="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# ---- 推理参数 ----
# 关键：训练数据是 512x512，推理必须匹配训练分辨率，用 1024 会导致 flow 错乱
IMG_SIZE=512
INFER_STEPS=50
RESIZE_MODE="stretch"
FLOW_ITERS=4   # 与训练 iters 对齐（diagnose.md Step 2）

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

$PYTHON qwen_image_flow_v4.py \
    --input_dir         "$INPUT_DIR" \
    --output_dir        "${OUTPUT_DIR}/${EXP_SUFFIX}" \
    --ckpt_path         "$CKPT" \
    --dit_target_layers "$DIT_LAYERS" \
    --lora_path         "$LORA_PATH" \
    --img_size          $IMG_SIZE \
    --infer_steps       $INFER_STEPS \
    --resize_mode       $RESIZE_MODE \
    --flow_iters        $FLOW_ITERS \
    --prompt            "$PROMPT"
