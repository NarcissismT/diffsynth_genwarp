#!/usr/bin/env bash
# =============================================================================
# Unified 统一模型推理示例（对应 scripts/train_unified.sh 训出的 checkpoint）
#
# 推理流程（单张 warped 图输入，无需 guide）：
#   warped → Stage-A prior 粗流 → 预矫正
#   → 模型内部 Qwen 抽 token（output_type=latent，不 decode RGB）
#   → 可靠性门控 + RAFT-like refiner 残差 → 几何复合 R(x)+B(x+R(x))
#   → 最终 backward flow → grid_sample 采样 warped 原图像素
#
# 每张图输出：
#   *_rectified.png          统一模型矫正结果（只从原图采样）
#   *_prior_rectified.png    同一次前向的 Stage-A 粗矫正（判断 joint 是否真有增益）
#   *_backward_flow.npy       后向光流
#   *_feature_confidence.png  Qwen 特征可靠性门控热图
#   *_valid.png / *_metadata.json
#
# 运行环境：容器 registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers
# （已预装 diffusers==0.35.2）。需大显存卡（加载 39GB Qwen transformer）。
# 参照 versions/.../scripts/flow_v4_4_sample.sh。
# =============================================================================
set -euo pipefail

# ---- 路径配置 ----
CONFIG="configs/unified.yaml"              # work_size=512、qwen 指向本地模型
CHECKPOINT="runs/d2r/unified/epoch_0008.pt"  # 想看别的 epoch 改这里（如 latest.pt）
INPUT_DIR="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical"  # 真实困难测试集（22 张高清）
OUTPUT_DIR="runs/d2r/unified/infer_typical_epoch0008"
STAGE="unified"

# ---- 切到仓库根目录 ----
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

# ---- 容器内用系统 python（判别方式同 flow_v4_4_sample.sh）----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    echo "[warn] 未检测到容器内 torch；确认用的是 diffsynth:v2-diffusers 镜像。" >&2
    PYTHON="/usr/bin/python"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[info] checkpoint=$CHECKPOINT input=$INPUT_DIR out=$OUTPUT_DIR"
echo "[info] 批处理模式：39GB Qwen 只加载一次，循环处理 $INPUT_DIR 下所有图。"

# ---- 批处理推理（Qwen 加载一次，循环所有图）----
"$PYTHON" -m diffusion2raft.infer \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --stage "$STAGE"

echo "[done] 结果在 $OUTPUT_DIR"
echo "对比看：<stem>_rectified.png（统一模型）vs <stem>_prior_rectified.png（仅 prior）"
echo "        bad2 是真实测试集，无 GT；看矫正结果的页面边界/行方向/文字是否变直。"
echo "        <stem>_feature_confidence.png 看 Qwen 特征在图上哪些区域被采信。"
# 单张推理：把 --input-dir 换成 --warped path/to/one.png
