INPUT_DIR="/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/bad2/表格线_101790.jpg"
OUTPUT_DIR="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/flow_warp_results"
LORA_PATH="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_ckpts/20260512-1_1in10_w_flow_head_v2/step-144000.safetensors"

/usr/bin/python qwen_image_flow_warp.py \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --lora_path   "$LORA_PATH" \
    --prompt "Flatten this warped or curled document image to a flat, undistorted version. Preserve all text, lines, and content accurately." \
    --img_size 1024 --infer_steps 50 --resize_mode stretch \
    --raft_model_size large --num_flow_updates 20 --batch_size 1
