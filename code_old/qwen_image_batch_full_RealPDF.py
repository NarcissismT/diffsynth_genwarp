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
from diffsynth import load_state_dict
import torchvision

def process_images(input_dir, output_dir, dit_path, prompt, batch_size=1):
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
    print(f"加载DIT模型: {dit_path}")
    state_dict = load_state_dict(dit_path)
    pipe.dit.load_state_dict(state_dict)
    
    # 查找所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    # 递归遍历所有子文件夹
    for root, _, _ in os.walk(input_dir):
        for ext in image_extensions:
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
                orig_img = img.copy()  # 保存原始图像用于创建对比图
                width, height = img.size
                
                print(f"处理图片: {rel_path}, 原始尺寸: {width}x{height}")
                
                # 计算短边resize到2048的缩放比例
                short_side = min(width, height)
                scale = 2048 / short_side
                # 按比例缩放图像
                new_height = round(height * scale)
                new_width = round(width * scale)
                
                img_resized = torchvision.transforms.functional.resize(
                    img,
                    (new_height, new_width),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )
                
                # 保存一份缩放后的图像供处理前查看
                img_resized_copy = img_resized.copy()
                
                # 中心裁剪到目标尺寸
                img_cropped = torchvision.transforms.functional.center_crop(img_resized, (1024, 1024))
                
                print(f"缩放至: {new_width}x{new_height}, 裁剪至: 1024x1024")
                
                # 处理裁剪后的图片
                enhanced_img = pipe(
                    prompt=prompt,
                    edit_image=img_cropped,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=50,
                    width=1024,
                    height=1024
                )
                
                # 创建对比图：原始裁剪图 + 增强后的图
                compare_img = Image.new('RGB', (2048, 1024))
                compare_img.paste(img_cropped, (0, 0))
                compare_img.paste(enhanced_img, (1024, 0))
                
                # 保存对比图
                compare_img.save(output_path)
                print(f"保存到: {output_path}")
                
                # # 同时保存一张完整的原图与处理图的对比图
                # # 为了更清楚地看到处理前后差异，我们保存原图、缩放图和裁剪+处理图
                # full_compare_path = os.path.splitext(output_path)[0] + "_full_compare.jpg"
                
                # # 创建一个包含三张图的对比图：原图、缩放后图和裁剪区域标记图
                # full_compare = Image.new('RGB', (width * 3, height))
                # full_compare.paste(orig_img, (0, 0))  # 第一列：原图
                
                # # 第二列：缩放后的图（需要缩放回原始大小）
                # resized_back = img_resized_copy.resize((width, height))
                # full_compare.paste(resized_back, (width, 0))
                
                # # 第三列：在缩放图上标记出裁剪区域
                # marked_img = resized_back.copy()
                
                # # 计算裁剪区域在缩放图上的位置
                # crop_left = (new_width - 1024) // 2
                # crop_top = (new_height - 1024) // 2
                # crop_right = crop_left + 1024
                # crop_bottom = crop_top + 1024
                
                # # 将裁剪区域在缩放回原尺寸后的图上标记出来
                # scale_back = width / new_width
                # mark_left = int(crop_left * scale_back)
                # mark_top = int(crop_top * scale_back)
                # mark_right = int(crop_right * scale_back)
                # mark_bottom = int(crop_bottom * scale_back)
                
                # # 创建一个临时画布来画红框
                # import PIL.ImageDraw as ImageDraw
                # draw = ImageDraw.Draw(marked_img)
                # draw.rectangle([mark_left, mark_top, mark_right, mark_bottom], outline="red", width=5)
                
                # # 将标记了裁剪区域的图放在第三列
                # full_compare.paste(marked_img, (width * 2, 0))
                
                # full_compare.save(full_compare_path)
                # print(f"保存完整对比图到: {full_compare_path}")
                
            except Exception as e:
                print(f"处理 {rel_path} 时出错: {str(e)}")
                # 记录错误到文件
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{rel_path}: {str(e)}\n")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用Qwen模型批量处理图片')
    parser.add_argument('--dit_path', type=str, required=True, help='模型路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片目录')
    parser.add_argument('--prompt', type=str, default='将这张图片变高清，去除反光，均衡光影', help='处理图片使用的提示词')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    
    args = parser.parse_args()
    
    # 批量处理图片
    process_images(args.input_dir, args.output_dir, args.dit_path, args.prompt, args.batch_size)