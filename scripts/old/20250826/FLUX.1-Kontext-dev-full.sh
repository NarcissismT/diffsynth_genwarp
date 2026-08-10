accelerate launch --config_file examples/flux/model_training/full/accelerate_config.yaml examples/flux/model_training/train.py \
  --dataset_base_path /juicefs-algorithm/data/IPT/junle_liu/滤镜/train/data/highlight_compareyuyi \
  --dataset_metadata_path /juicefs-algorithm/data/IPT/junle_liu/滤镜/train/data/highlight_compareyuyi/metadata_highlight_compareyuyi_kontext.csv \
  --data_file_keys "image,kontext_images" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/FLUX.1-Kontext-dev_lora/20250826/dehighlight_compareyuyi" \
  --trainable_models "dit" \
  --extra_inputs "kontext_images" \
  --use_gradient_checkpointing \
  --save_steps 200 \
  --model_paths '[
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors",
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/text_encoder/model.safetensors",
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/text_encoder_2",
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors"
]' \