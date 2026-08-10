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
import csv

import diffsynth
print("Using diffsynth from:", diffsynth.__file__)


def process_images(input_dir, output_dir, dit_path, lora_path, prompt, batch_size=1, crop=True, prompt_dict=None, short_side_pixel=2048, img_size=1024, enable_dit_fp8_computation=False, total_divide=1, divide_index=0, infer_steps=50, resize_mode = "stretch"):
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
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
            ]),
            ModelConfig([
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00001-of-00004.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00002-of-00004.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00003-of-00004.safetensors",
                "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/text_encoder/model-00004-of-00004.safetensors"
            ]),
            ModelConfig("/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors")
        ],
        tokenizer_config=ModelConfig("/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/tokenizer"),
        processor_config=ModelConfig("/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit/processor"),
    )
    
    
    if dit_path:
        print(f"加载DIT模型: {dit_path}")
        state_dict = load_state_dict(dit_path)
        pipe.dit.load_state_dict(state_dict)

    if lora_path:
        print(f"加载LoRA模型: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path)

    
    pipe.enable_vram_management(enable_dit_fp8_computation=enable_dit_fp8_computation)

    if prompt_dict:
        prompt_map = {}
        with open(prompt_dict, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_map[row['image']] = row['prompt']
    else:
        prompt_map = None


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

        if total_divide > 1:
            # 计算当前批次所属的子文件夹编号
            current_divide_index = (i // batch_size) % total_divide
            if current_divide_index != divide_index:
                continue

        batch_files = image_files[i:i+batch_size]
        
        for img_path in batch_files:
            print(img_path)
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
                
                #!新的resize方式
                if resize_mode == "crop":
                    # 计算短边resize到2048的缩放比例
                    short_side = min(width, height)
                    scale = short_side_pixel / short_side
                    # 按比例缩放图像
                    new_height = round(height * scale)
                    new_width = round(width * scale)

                    img_resized = torchvision.transforms.functional.resize(
                        img,
                        (new_height, new_width),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    )

                    # img_resized_copy = img_resized.copy()

                    # 中心裁剪到目标尺寸
                    img_cropped = torchvision.transforms.functional.center_crop(img_resized, (img_size, img_size))
                    new_width, new_height = img_size, img_size

                elif resize_mode == "stretch":
                    # 不裁剪，直接调整到img_size x img_size（可能会改变宽高比）
                    img_cropped = img.resize((img_size, img_size), Image.LANCZOS)
                    new_width, new_height = img_size, img_size

                elif resize_mode == "scale_to_short_side":
                    # 将短边缩小到 img_size，保持宽高比
                    short_side = min(width, height)
                    scale = img_size / short_side
                    scaled_height = int(height * scale)
                    scaled_width = int(width * scale)

                    # 向下取整为16的倍数
                    target_height = scaled_height // 16 * 16
                    target_width = scaled_width // 16 * 16

                    img_cropped = torchvision.transforms.functional.resize(
                        img,
                        (target_height, target_width),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    )
                    new_width, new_height = target_width, target_height


                print(f"缩放至: {new_width}x{new_height}")

                if prompt_map:
                    prompt = prompt_map[os.path.basename(img_path)]
                
                # pipeline处图片
                enhanced_img = pipe(
                    prompt=prompt,
                    edit_image=img_cropped,
                    seed=random.randint(0, 2**32 - 1),
                    num_inference_steps=infer_steps,
                    width=new_width,
                    height=new_height
                    # width=img_size,
                    # height=img_size
                )

                
                # 假设 output_path 是基础路径，例如 "/your/output/folder/sample001.jpg"
                # 我们将其扩展为三张图的保存路径：
                output_path_a = output_path.replace(".jpg", "_a.jpg")
                output_path_b = output_path.replace(".jpg", "_b.jpg")
                output_path_c = output_path.replace(".jpg", "_c.jpg")

                # 创建对比图
                compare_img = Image.new('RGB', (2 * img_size, img_size))
                compare_img.paste(img_cropped, (0, 0))
                compare_img.paste(enhanced_img, (img_size, 0))

                # 保存三张图像
                img_cropped.save(output_path_a)
                enhanced_img.save(output_path_b)
                compare_img.save(output_path_c)

                print(f"已保存裁剪图像到: {output_path_a}")
                print(f"已保存增强图像到: {output_path_b}")
                print(f"已保存对比图像到: {output_path_c}")

                
                
            except Exception as e:
                print(f"处理 {rel_path} 时出错: {str(e)}")
                # 记录错误到文件
                with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                    f.write(f"{rel_path}: {str(e)}\n")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用Qwen模型批量处理图片')
    parser.add_argument('--dit_path', type=str, help='模型路径')
    parser.add_argument('--lora_path', type=str, help='LoRA模型路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片目录')
    parser.add_argument('--prompt', type=str, help='处理图片使用的提示词')
    parser.add_argument('--prompt_dict', type=str, help='处理图片使用的提示词字典路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--short_side_pixel', type=int, default=1024, help='短边缩放到的像素值')
    parser.add_argument('--img_size', type=int, default=1024, help='输出图片的尺寸')
    parser.add_argument('--total_divide', type=int, default=1, help='总共分成多少份')
    parser.add_argument('--divide_index', type=int, default=0, help='子文件夹编号')
    parser.add_argument('--infer_steps', type=int, default=50, help='推理步数')
    parser.add_argument('--enable_dit_fp8_computation', action='store_true', help='是否启用DIT的FP8计算以节省显存')
    parser.add_argument('--crop', action='store_true', help='是否裁切')
    parser.add_argument('--resize_mode', type = str, default="stretch", help='resize方式')

    
    args = parser.parse_args()

    print('crop', args.crop)
    
    # 批量处理图片
    process_images(args.input_dir, args.output_dir, args.dit_path, args.lora_path, args.prompt, args.batch_size, args.crop, args.prompt_dict, args.short_side_pixel, args.img_size, args.enable_dit_fp8_computation, args.total_divide, args.divide_index,args.infer_steps, args.resize_mode)