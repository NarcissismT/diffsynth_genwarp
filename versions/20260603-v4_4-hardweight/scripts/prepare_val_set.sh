#!/usr/bin/env bash
# =============================================================================
# 验证集准备（一次性，两步）— 给带验证的训练用
#   step 1: make_val_split.py        按难度分层切 held-out（产出 metadata_train/val.csv）
#   step 2: gen_val_corrected_low.py 对 val 集预生成真扩散 corrected_low（产出 val_index.csv）
#
# 跑法：
#   CUDA_VISIBLE_DEVICES=<空闲卡> bash scripts/prepare_val_set.sh
# 完成后用 scripts/train_v4_4_layer36_val.sh 开训（自动带验证）。
# =============================================================================
set -e

DATA_DIR=/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511
FULL_CSV=$DATA_DIR/metadata_with_vae.csv
VAL_CSV=$DATA_DIR/metadata_val.csv
VAL_LOW_DIR=$DATA_DIR/val_corrected_low
LORA="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"

NUM_VAL=300        # 验证集总量（分 low/mid/high 三层各 100）
NUM_PROBE=1500     # 难度分层候选池

if [ -d /usr/local/lib/python3.10/dist-packages/torch ]; then
    PYTHON="/usr/bin/python"
else
    PYTHON="/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python"
fi

cd "$(dirname "$0")/.."

echo "==== step 1/2: 按难度分层划分 held-out 验证集 ===="
$PYTHON make_val_split.py \
    --csv "$FULL_CSV" \
    --num_val $NUM_VAL --num_probe $NUM_PROBE --seed 42

echo; echo "==== step 2/2: 预生成验证集真扩散 corrected_low（Qwen 50 步）===="
$PYTHON gen_val_corrected_low.py \
    --val_csv "$VAL_CSV" \
    --out_dir "$VAL_LOW_DIR" \
    --lora_path "$LORA" \
    --img_size 512 --resize_mode stretch --infer_steps 50 --dit_target_layers 35

echo; echo "✅ 验证集准备完成："
echo "   训练集 CSV: $DATA_DIR/metadata_train.csv"
echo "   验证索引:   $VAL_LOW_DIR/val_index.csv"
echo "   开训: bash scripts/train_v4_4_layer36_val.sh"
