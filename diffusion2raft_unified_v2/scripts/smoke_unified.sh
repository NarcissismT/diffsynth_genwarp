#!/usr/bin/env bash
# =============================================================================
# 1-batch 真 Qwen 冒烟：加载 39GB Qwen 权重、跑真前向，验证 diffusers 0.35.2
# 抽出的 transformer block 输出格式和 unified 代码 hook 假设一致。
#
# 必须在大显存卡上跑（A100/H100 80GB）；A10 的 23GB 装不下 39GB transformer。
# 容器用 registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers（已预装 diffusers）。
#
# 只跑 4 张训练 / 2 张验证、1 epoch。通过后再用 train_unified.sh 跑完整 20 epoch。
# =============================================================================
set -euo pipefail

CONFIG="configs/unified_smoke.yaml"
STAGE="unified"
RESUME="runs/d2r/prior/latest.pt"
MASTER_PORT="${MASTER_PORT:-29532}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

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

# 冒烟固定单进程单卡（1 张大显存卡即可），避免多卡通信干扰问题定位。
echo "[info] 1-batch Qwen 冒烟：config=$CONFIG resume=$RESUME python=$PYTHON"
echo "[info] 需要单张 A100/H100（39GB transformer + 激活）。"

"$PYTHON" -m diffusion2raft.train \
    --config "$CONFIG" \
    --stage "$STAGE" \
    --resume "$RESUME" \
    --epochs 1

# =============================================================================
# 期望看到：
#   - Qwen pipeline 加载成功（无 KeyError / shape 报错）
#   - epoch=1 step=1..4 有 loss、epe、prior_epe、feature_confidence 打印
#   - validation epoch=1 出 epe / prior_epe / fold_rate
#   - saved checkpoint: runs/d2r_smoke/unified/latest.pt
#
# 若报 "Qwen transformer block output changed; expected (text_hidden, image_hidden)"
# 或 "no source-condition tokens" —— 说明 diffusers 0.35.2 的 block 输出格式和
# unified 代码假设不一致，需要按实际输出调整 models/unified.py 的 _capture_one。
# 把该报错贴给我即可，我据实际 tensor 形状修 hook。
# =============================================================================
