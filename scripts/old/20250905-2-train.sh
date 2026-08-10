csv_path=data/train/pdf20250727/metadata_矫正.csv
train_id=20250905-2

CUDA_VISIBLE_DEVICES=6 accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path $(dirname "$csv_path") \
  --dataset_metadata_path $csv_path \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/$train_id" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --tokenizer_path "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/tokenizer" \
  --processor_path "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/processor" \
  --model_paths '[
    [
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
    ],
    [
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00001-of-00004.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00002-of-00004.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00003-of-00004.safetensors",
        "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00004-of-00004.safetensors"
    ],
    "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors"
]' \
