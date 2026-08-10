
import os
import torch
import glob
from tqdm import tqdm
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import random

def ensure_multiple_of_n(value, n=16):
    """确保值是n的倍数"""
    return ((value + n - 1) // n) * n

def process_images(input_dir, output_dir, prompt, batch_size=1):
    """批量处理图片并保持原始宽高"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    print("正在加载模型...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig([
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
            ]),
            ModelConfig([
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00001-of-00004.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00002-of-00004.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00003-of-00004.safetensors",
                "/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/text_encoder/model-00004-of-00004.safetensors"
            ]),
            ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors")
        ],
        tokenizer_config=ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/tokenizer"),
        processor_config=ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/Qwen-Image-Edit/processor"),
    )
    
    # 加载LoRA模型
    pipe.load_lora(pipe.dit, "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/models/train/Qwen-Image-Edit_lora/20250821/highlight-new/step-250.safetensors")
    
    # 查找所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 检查已处理过的图片
    processed_files = set(os.listdir(output_dir))
    
    # 分批处理图片
    total_batches = (len(image_files) + batch_size - 1) // batch_size  # 向上取整
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, len(image_files), batch_size), total=total_batches, desc="处理图片批次"):
        batch_files = image_files[i:i+batch_size]
        
        for img_path in batch_files:
            # 获取文件名
            base_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, base_name)
            
            # 检查是否已经处理过
            if base_name in processed_files:
                print(f"跳过已处理的图片: {base_name}")
                continue
            
            try:
                # 打开图片并获取原始尺寸
                img = Image.open(img_path)
                orig_width, orig_height = img.size
                
                # 调整宽高为16的倍数
                adj_width = ensure_multiple_of_n(orig_width, 16)
                adj_height = ensure_multiple_of_n(orig_height, 16)
                
                if adj_width != orig_width or adj_height != orig_height:
                    print(f"调整图片尺寸: {orig_width}x{orig_height} -> {adj_width}x{adj_height}")
                    # # 使用LANCZOS重采样确保高质量调整大小
                    img = img.resize((1024,1024))
                
                # 处理图片
                print(f"处理图片: {base_name}, 尺寸: {adj_width}x{adj_height}")
                enhanced_img = pipe(
                    prompt=prompt,
                    edit_image=img,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=50,
                )
                
                # 如果尺寸曾经调整过，恢复到原始尺寸
                # if adj_width != orig_width or adj_height != orig_height:
                enhanced_img = enhanced_img.resize((orig_width, orig_height))
                
                # 保存结果
                enhanced_img.save(output_path)
                print(f"保存到: {output_path}")
                
            except Exception as e:
                print(f"处理 {base_name} 时出错: {str(e)}")
                # 记录错误到文件
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{base_name}: {str(e)}\n")

if __name__ == "__main__":
    # 配置参数
    input_directory = "/juicefs-algorithm/data/IPT/junle_liu/poster_dataset/dataset/canva/poster"
    output_directory = "/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/results/canva_poster/highlight-new-step250"
    
    # 定义prompt
    prompt_text = "请给以下图片随机加入高光"
    
    # 批量处理图片
    process_images(input_directory, output_directory, prompt_text)
