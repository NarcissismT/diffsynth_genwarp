#!/usr/bin/env bash
# =============================================================================
# 6 个 layer probing 实验的定量对比脚本
# =============================================================================
#
# 用法：
#   bash scripts/eval_exp_compare.sh [合成|真实|both]
#
#   合成模式（默认）：在合成验证集上算 EPE + WarpL1，需要有 flow_gt
#   真实模式：在 silver_bullet 真实图上跑推理 + 保存对比图（无 EPE）
#   both：同时跑两种模式
#
# 前置条件：
#   - 6 个实验的训练已完成，各自有 step-XXXXX.safetensors
#   - 修改下方 CKPT_XXX 变量为各实验最新/最优 checkpoint 路径
# =============================================================================

MODE="${1:-合成}"

EVAL_RESULT_DIR="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts/eval_results"
VIS_BASE_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/eval_vis"
REAL_VAL_DIR="/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/dewarp"
LORA_CKPT="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

# ---- 各实验 checkpoint（训练完成后，填入对应最优 step）----
# 格式：CKPT_<实验名>=<路径>，留空表示跳过该实验
CKPT_BASE="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_ckpts"

CKPT_A="${CKPT_BASE}/20260515-exp_A_layer12/step-REPLACE.safetensors"
CKPT_B="${CKPT_BASE}/20260515-exp_B_layer24/step-REPLACE.safetensors"
CKPT_C="${CKPT_BASE}/20260515-exp_C_layer36/step-REPLACE.safetensors"
CKPT_D="${CKPT_BASE}/20260515-exp_D_layer48/step-REPLACE.safetensors"
CKPT_E="${CKPT_BASE}/20260515-exp_E_layer24_36_48/step-REPLACE.safetensors"
CKPT_F="${CKPT_BASE}/20260515-exp_F_layer12_24_36_48/step-REPLACE.safetensors"

# ---- Python 路径 ----
if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

cd "$(dirname "$0")/.."
mkdir -p "$EVAL_RESULT_DIR"

# =============================================================================
# 辅助函数：跑单个实验评测
# =============================================================================

run_eval_synthetic() {
    local exp_name=$1
    local ckpt_path=$2
    local dit_layers=$3
    local result_json="${EVAL_RESULT_DIR}/${exp_name}_synthetic.json"

    if [ ! -f "$ckpt_path" ]; then
        echo "[跳过] ${exp_name}: ckpt 不存在（${ckpt_path}）"
        return
    fi

    echo ""
    echo ">>> [合成评测] ${exp_name}  layers=${dit_layers}"
    $PYTHON eval_flow_v4_quant.py \
        --ckpt_path          "$ckpt_path" \
        --dit_target_layers  "$dit_layers" \
        --lora_checkpoint    "$LORA_CKPT" \
        --exp_name           "$exp_name" \
        --result_json        "$result_json" \
        --per_category       100 \
        --flow_iters         12
}

run_eval_real() {
    local exp_name=$1
    local ckpt_path=$2
    local dit_layers=$3
    local result_json="${EVAL_RESULT_DIR}/${exp_name}_real.json"
    local vis_dir="${VIS_BASE_DIR}/${exp_name}"

    if [ ! -f "$ckpt_path" ]; then
        echo "[跳过] ${exp_name}: ckpt 不存在（${ckpt_path}）"
        return
    fi

    echo ""
    echo ">>> [真实图评测] ${exp_name}  layers=${dit_layers}"
    $PYTHON eval_flow_v4_quant.py \
        --ckpt_path          "$ckpt_path" \
        --dit_target_layers  "$dit_layers" \
        --lora_checkpoint    "$LORA_CKPT" \
        --lora_path          "$LORA_CKPT" \
        --real_val_dir       "$REAL_VAL_DIR" \
        --vis_output_dir     "$vis_dir" \
        --max_real_samples   200 \
        --exp_name           "$exp_name" \
        --result_json        "$result_json" \
        --flow_iters         12 \
        --img_size           1024 \
        --infer_steps        50 \
        --resize_mode        stretch
}

# =============================================================================
# 跑各实验
# =============================================================================

run_exp() {
    local mode=$1
    if [ "$mode" = "合成" ] || [ "$mode" = "both" ]; then
        run_eval_synthetic "exp_A_layer12"         "$CKPT_A" "11"
        run_eval_synthetic "exp_B_layer24"         "$CKPT_B" "23"
        run_eval_synthetic "exp_C_layer36"         "$CKPT_C" "35"
        run_eval_synthetic "exp_D_layer48"         "$CKPT_D" "47"
        run_eval_synthetic "exp_E_layer24_36_48"   "$CKPT_E" "23,35,47"
        run_eval_synthetic "exp_F_layer12_24_36_48" "$CKPT_F" "11,23,35,47"
    fi
    if [ "$mode" = "真实" ] || [ "$mode" = "both" ]; then
        run_eval_real "exp_A_layer12"         "$CKPT_A" "11"
        run_eval_real "exp_B_layer24"         "$CKPT_B" "23"
        run_eval_real "exp_C_layer36"         "$CKPT_C" "35"
        run_eval_real "exp_D_layer48"         "$CKPT_D" "47"
        run_eval_real "exp_E_layer24_36_48"   "$CKPT_E" "23,35,47"
        run_eval_real "exp_F_layer12_24_36_48" "$CKPT_F" "11,23,35,47"
    fi
}

run_exp "$MODE"

# =============================================================================
# 汇总打印对比表格
# =============================================================================

echo ""
echo "=============================="
echo " 汇总结果"
echo "=============================="

if [ "$MODE" = "合成" ] || [ "$MODE" = "both" ]; then
    SYNTHETIC_JSONS=$(ls "${EVAL_RESULT_DIR}"/*_synthetic.json 2>/dev/null | tr '\n' ',')
    if [ -n "$SYNTHETIC_JSONS" ]; then
        $PYTHON eval_flow_v4_quant.py \
            --ckpt_path dummy --dit_target_layers "0" \
            --compare "${SYNTHETIC_JSONS%,}"
    fi
fi

if [ "$MODE" = "真实" ] || [ "$MODE" = "both" ]; then
    REAL_JSONS=$(ls "${EVAL_RESULT_DIR}"/*_real.json 2>/dev/null | tr '\n' ',')
    if [ -n "$REAL_JSONS" ]; then
        $PYTHON eval_flow_v4_quant.py \
            --ckpt_path dummy --dit_target_layers "0" \
            --compare "${REAL_JSONS%,}"
    fi
    echo ""
    echo "对比图目录: ${VIS_BASE_DIR}"
    echo "  每个实验的对比图在对应子目录中（原始 | warp 结果 并排）"
fi
