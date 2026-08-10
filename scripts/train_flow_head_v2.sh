csv_path=/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white/metadata_with_flow.csv
train_id=20260513-1_flow_head_v2

accelerate launch --config_file scripts/Acceconfig_8A800.yaml train_flow_head_v2.py \
  --dataset_metadata_path $csv_path \
  --output_path "/juicefs-algorithm/data/IPT/zhuochu_yang/flow_head_v2_ckpts/$train_id" \
  --train_size     512 \
  --batch_size     4 \
  --num_steps      50000 \
  --learning_rate  2e-4 \
  --weight_decay   1e-5 \
  --gamma          0.8 \
  --lambda_warp    0.0 \
  --iters_train    4 \
  --num_workers    4 \
  --save_steps     4000 \
  --log_steps      100
