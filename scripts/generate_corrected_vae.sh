INPUT_CSV="/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_flow.csv"
OUTPUT_DIR="/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511"
VAE_IMG_DIR="$OUTPUT_DIR/corrected_vae"

# 使用 SLURM 任务 ID 作为分片索引（每次提交一个 job，传入不同的 DIVIDE_INDEX）
# 用法：TOTAL_DIVIDE=8 DIVIDE_INDEX=0 bash scripts/generate_corrected_vae.sh
TOTAL_DIVIDE=${TOTAL_DIVIDE:-10}
DIVIDE_INDEX=${DIVIDE_INDEX:-0}

echo "分片 $DIVIDE_INDEX / $TOTAL_DIVIDE"

/usr/bin/python utils/generate_corrected_vae.py \
    --input_csv    "$INPUT_CSV" \
    --output_csv   "$OUTPUT_DIR/metadata_with_corrected_vae_part${DIVIDE_INDEX}.csv" \
    --vae_img_dir  "$VAE_IMG_DIR" \
    --batch_size   36 \
    --total_divide $TOTAL_DIVIDE \
    --divide_index $DIVIDE_INDEX
