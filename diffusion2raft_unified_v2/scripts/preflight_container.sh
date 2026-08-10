#!/usr/bin/env bash
# =============================================================================
# 容器内预检：在提交完整训练前，验证 unified 阶段唯一未知项 ——
# 容器里的 diffusers 是否带 QwenImageEditPipeline，且本地 Qwen 目录能否
# 用 from_pretrained 加载。参考脚本用的是 DiffSynth 的 ModelManager，而
# unified_v2 走 diffusers from_pretrained，两者不是同一条加载路径。
#
# 在容器内跑：  bash scripts/preflight_container.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd -P)/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PYTHON="/usr/bin/python"
[ -x "$PYTHON" ] || PYTHON="python"

"$PYTHON" - <<'PY'
import sys
ok = True

# 1. diffusers 版本 + Qwen 管线类
try:
    import diffusers
    print("diffusers:", diffusers.__version__)
    from diffusers import QwenImageEditPipeline
    print("QwenImageEditPipeline: OK")
    try:
        from diffusers import QwenImageEditPlusPipeline
        print("QwenImageEditPlusPipeline: OK")
    except Exception as e:
        print("QwenImageEditPlusPipeline: MISSING (base Qwen-Image-Edit 仍可用) ->", e)
except Exception as e:
    ok = False
    print("diffusers/Qwen import FAILED:", type(e).__name__, e)

# 2. 本地模型是否 diffusers 格式
import os
mid = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"
print("model_index.json:", "OK" if os.path.exists(os.path.join(mid, "model_index.json")) else "MISSING")

# 3. 统一图（lite backend，不加载 Qwen）能否构建 + 前向
try:
    import torch
    from diffusion2raft.models import build_rectifier
    m = build_rectifier(
        {"feature_backend": "lite", "feature_channels": 32, "prior_base_channels": 32,
         "cnn_feature_channels": 24, "refiner_hidden_channels": 32,
         "refiner_iterations": 2, "correlation_radius": 2},
        {}, stage="unified", device="cpu",
    )
    out = m(torch.rand(1, 3, 128, 128), None, stage="unified")
    print("unified lite forward: OK, final_flow", tuple(out["final_flow"].shape))
except Exception as e:
    ok = False
    print("unified graph FAILED:", type(e).__name__, e)

# 4.（可选，重）尝试真正加载 Qwen transformer —— 只在前面都 OK 时提示
print("\n下一步：若上面全 OK，可用 1-batch 真 Qwen 前向确认（会加载 39GB，需大显存）：")
print("  bash scripts/train_unified.sh  # 先把 configs/unified.yaml 的 epochs 设 1 做冒烟")
sys.exit(0 if ok else 1)
PY
