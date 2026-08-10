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
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_v4_4_results"

# ---- FlowHead V4.4 checkpoint（path B + 难度加权 + 梯度累积 + VAE域对齐，单层 Layer 36）----
# 注意：带 VAE 域对齐的训练输出在 20260608 文件夹（20260603 是修复前旧训练）
CKPT="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_4_ckpts/20260610-v4_4-vaeroundtrip-layer36/step-435060.safetensors"
DIT_LAYERS="35"
K_SOURCE="warped"        # 必须与训练一致（pathB=warped）
EXP_SUFFIX="v4_4_step435k_trainprompt"   # 用训练同款长 prompt 重测（对比 _0610_step435k 短prompt炸裂版）

# ---- DiT LoRA（与训练一致）----
LORA_PATH="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# ---- 推理参数 ----
IMG_SIZE=512        # 必须与训练分辨率一致
INFER_STEPS=50
RESIZE_MODE="stretch"
FLOW_ITERS=4        # 与训练 iters=4 对齐

# ---- Prompt ----
# ★★★ 必须与训练完全一致（2026-06-15 根因）：训练全量 11.6 万样本都用下面这条长 prompt。
#   prompt 经 text cross-attention 改变 layer-35 的 Q/K；FlowHead 过拟合了训练 Q/K 分布，
#   换 prompt → Q/K 偏离训练分布 → flow 塌进 ~100px 垃圾吸引子（推理炸裂 151px 的主因之一）。
#   旧短 prompt "Flatten this warped..." 模型从未见过，已废弃。
PROMPT="Apply geometric correction to the input image to eliminate distortions such as skew, curl, folds, or non-frontal perspective, producing a flat, front-facing image where all textual content and structural elements, including table lines, are strictly aligned horizontally or vertically, resembling the layout of a standard PDF document. Preserve all original document content without alteration or regeneration. Retain all foreground elements, including any finger occlusions or shadows. If geometric correction creates undefined or empty regions at the edges, fill those areas with solid white pixels."

# ---- Python 解释器 ----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

# 切到本脚本所在版本目录运行（import 本地 utils）
cd "$(dirname "$0")/.."

$PYTHON qwen_image_flow_v4_4.py \
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
