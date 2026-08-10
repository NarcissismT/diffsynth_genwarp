CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path /juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/highlight/ \
  --dataset_metadata_path /juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/data/highlight/metadata_edit_kontext_1.csv \
  --data_file_keys "image,kontext_images" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/lora/FLUX.1-Kontext-dev_lora/highlight" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --align_to_opensource_format \
  --extra_inputs "kontext_images" \
  --use_gradient_checkpointing \
  --save_steps 1000 \
  --model_paths '[
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors",
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/text_encoder/model.safetensors",
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/text_encoder_2",
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors"
]' \
