#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import glob
import argparse
from tqdm import tqdm
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import random

def ensure_multiple_of_n(value, n=16):
    """确保值是n的倍数"""
    return ((value + n - 1) // n) * n

def process_images(input_dir, output_dir, lora_path, prompt, batch_size=1):
    """批量处理图片并保持原始宽高，支持子文件夹"""
    
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
    print(f"加载LoRA模型: {lora_path}")
    pipe.load_lora(pipe.dit, lora_path)
    
    # 收集所有图片文件(包括子文件夹)
    image_files = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    # 递归遍历所有子文件夹
    for root, _, _ in os.walk(input_dir):
        for ext in image_extensions:
            # 使用相对路径，以便后面创建输出路径时保持目录结构
            files = glob.glob(os.path.join(root, ext))
            files.extend(glob.glob(os.path.join(root, ext.upper())))
            image_files.extend(files)
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 分批处理图片
    total_batches = (len(image_files) + batch_size - 1) // batch_size  # 向上取整
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, len(image_files), batch_size), total=total_batches, desc="处理图片批次"):
        batch_files = image_files[i:i+batch_size]
        
        for img_path in batch_files:
            # 获取相对于输入目录的相对路径
            rel_path = os.path.relpath(img_path, input_dir)
            # 构建输出路径，保持相同的目录结构
            output_path = os.path.join(output_dir, rel_path)
            # 确保输出文件的目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 检查是否已经处理过
            if os.path.exists(output_path):
                print(f"跳过已处理的图片: {rel_path}")
                continue
            
            try:
                # 打开图片并转换为RGB模式，去除Alpha通道
                img = Image.open(img_path).convert('RGB')
                orig_width, orig_height = img.size
                
                # 调整宽高为16的倍数
                adj_width = ensure_multiple_of_n(orig_width, 16)
                adj_height = ensure_multiple_of_n(orig_height, 16)
                
                if adj_width != orig_width or adj_height != orig_height:
                    print(f"调整图片尺寸: {orig_width}x{orig_height} -> {adj_width}x{adj_height}")
                    # 使用LANCZOS重采样确保高质量调整大小
                    img = img.resize((1024, 1024))
                
                # 处理图片
                print(f"处理图片: {rel_path}, 尺寸: {img.size[0]}x{img.size[1]}")
                enhanced_img = pipe(
                    prompt=prompt,
                    edit_image=img,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=50,
                    width=1024,
                    height=1024
                )
                
                # 如果尺寸曾经调整过，恢复到原始尺寸
                enhanced_img = enhanced_img.resize((orig_width, orig_height))
                
                # 保存结果
                enhanced_img.save(output_path)
                print(f"保存到: {output_path}")
                
            except Exception as e:
                print(f"处理 {rel_path} 时出错: {str(e)}")
                # 记录错误到文件
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{rel_path}: {str(e)}\n")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用Qwen模型批量处理图片')
    parser.add_argument('--lora_path', type=str, required=True, help='LoRA模型路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片目录')
    parser.add_argument('--prompt', type=str, default='将这张图片变高清，去除反光，均衡光影', help='处理图片使用的提示词')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    
    args = parser.parse_args()
    
    # 批量处理图片
    process_images(args.input_dir, args.output_dir, args.lora_path, args.prompt, args.batch_size)