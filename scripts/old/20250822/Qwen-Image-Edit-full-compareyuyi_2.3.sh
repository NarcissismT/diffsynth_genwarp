CUDA_VISIBLE_DEVICES=1,4,5,6,7,8,9 accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload_7.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path /juicefs-algorithm/data/IPT/junle_liu/滤镜/train/data/highlight_compareyuyi \
  --dataset_metadata_path /juicefs-algorithm/data/IPT/junle_liu/滤镜/train/data/highlight_compareyuyi/metadata_highlight_compareyuyi.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/result/full/qwen-image-edit/20250822/dehighlight_compareyuyi" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
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