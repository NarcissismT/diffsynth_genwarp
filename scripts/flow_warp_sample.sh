#!/bin/bash
# =============================================================================
# 两阶段光流 Warp 推理示例脚本
# 对应脚本: qwen_image_flow_warp.py
#
# 与 qwen_image_batch_full_lora.py 的差异：
#   - 新增 --raft_model_size / --num_flow_updates 控制光流模型
#   - 输出额外一张高清 warp 结果（c_*.jpg）和三合一对比图（d_*.jpg）
#   - 不传 --lora_path 时直接用基础模型推理
# =============================================================================

# ---- 路径配置（按实际情况修改）----
INPUT_DIR="vscode-remote://cloud-480010398094.ide.intsig.net/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp"
OUTPUT_DIR="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/dewarp_0515"
LORA_PATH="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# ---- 推理参数 ----
IMG_SIZE=1024            # 扩散推理分辨率
INFER_STEPS=50           # 去噪步数
RESIZE_MODE="stretch"    # stretch / crop / scale_to_short_side

# ---- 光流参数 ----
RAFT_SIZE="large"        # large（精度高）或 small（速度快）
FLOW_ITERS=20            # RAFT 迭代次数，越大越精确

# ---- Prompt ----
PROMPT="Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately."

# ---- Python 解释器（容器内用系统 python，宿主机用 RAFT_flow conda 环境）----
# 用容器内 torch 的安装路径来判断是否在容器内（宿主机无此路径）。
# 注意：pyxis/enroot 不创建 /.dockerenv，不能用它来判断。
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

# ---- 运行 ----
cd "$(dirname "$0")/.."

$PYTHON qwen_image_flow_warp.py \
    --input_dir        "$INPUT_DIR" \
    --output_dir       "$OUTPUT_DIR" \
    --lora_path        "$LORA_PATH" \
    --prompt           "$PROMPT" \
    --img_size         $IMG_SIZE \
    --infer_steps      $INFER_STEPS \
    --resize_mode      $RESIZE_MODE \
    --raft_model_size  $RAFT_SIZE \
    --num_flow_updates $FLOW_ITERS \
    --batch_size       1

# =============================================================================
# 可选：多 GPU 并行（分片处理）
# 在 N 台机器或 N 个进程上分别运行，每个进程处理不同的图片子集：
#
#   --total_divide N --divide_index 0   # 第 1 个进程
#   --total_divide N --divide_index 1   # 第 2 个进程
#   ...
#
# 可选：启用 FP8 推理节省显存（A100/H100）：
#   --enable_dit_fp8_computation
# =============================================================================
