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
import torchvision.transforms as T


def process_images(input_dir, output_dir, dit_path, lora_path, prompt, batch_size=1, crop=True, prompt_dict=None, short_side_pixel=2048, img_size=1024, enable_dit_fp8_computation=False):
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

    if dit_path:
        print(f"加载DIT模型: {dit_path}")
        state_dict = load_state_dict(dit_path)
        pipe.dit.load_state_dict(state_dict)

    if lora_path:
        print(f"加载LoRA模型: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path)

    pipe.enable_vram_management(enable_dit_fp8_computation=enable_dit_fp8_computation)


    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, input_dir, prompt_dict=None, prompt=None, img_size=1024, crop=True, short_side_pixel=2048, transform=None):
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

            self.image_files = image_files

            if prompt_dict:
                self.prompt_map = {}
                with open(prompt_dict, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.prompt_map[row['image']] = row['prompt']
            else:
                self.prompt_map = None

            self.prompt = prompt
            self.img_size = img_size
            self.crop = crop
            self.short_side_pixel = short_side_pixel
            self.transform = transform or T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
            ])

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            # 获取相对于输入目录的相对路径
            rel_path = os.path.relpath(img_path, input_dir)
            # 构建输出路径，保持相同的目录结构
            output_path = os.path.join(output_dir, rel_path)
            # 确保输出文件的目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 打开图片并转换为RGB模式，去除Alpha通道
            img = Image.open(img_path).convert('RGB')
            orig_img = img.copy()  # 保存原始图像用于创建对比图
            width, height = img.size

            print(f"处理图片: {rel_path}, 原始尺寸: {width}x{height}")

            if self.crop:
                # 计算短边resize到2048的缩放比例
                short_side = min(width, height)
                scale = self.short_side_pixel / short_side
                # 按比例缩放图像
                new_height = round(height * scale)
                new_width = round(width * scale)

                img_resized = torchvision.transforms.functional.resize(
                    img,
                    (new_height, new_width),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )

                # 中心裁剪到目标尺寸
                img_cropped = torchvision.transforms.functional.center_crop(img_resized, (self.img_size, self.img_size))

            else:
                # 不裁剪，直接调整到img_size x img_size（可能会改变宽高比）
                img_cropped = img.resize((self.img_size, self.img_size), Image.LANCZOS)

            # 使用转换，将图像转换为Tensor
            img_cropped = self.transform(img_cropped)

            # 获取提示词
            if self.prompt_map:
                prompt = self.prompt_map[os.path.basename(img_path)]
            else:
                prompt = self.prompt

            return img_cropped, orig_img, rel_path, output_path, prompt


    dataset = ImageDataset(input_dir, prompt_dict=prompt_dict, prompt=prompt, img_size=img_size, crop=crop, short_side_pixel=short_side_pixel)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 逐批处理图片
    for batch in tqdm(dataloader, desc="处理图片"):
        imgs_cropped, orig_imgs, rel_paths, output_paths, prompts = batch

        # 处理裁剪后的图片
        enhanced_img = pipe(
            prompt=prompt,
            edit_image=imgs_cropped,
            seed=random.randint(0, 2**32 - 1),
            num_inference_steps=50,
            width=img_size,
            height=img_size
        )

        # 创建对比图：原始裁剪图 + 增强后的图
        compare_img = Image.new('RGB', (2*img_size, img_size))
        compare_img.paste(orig_imgs[0], (0, 0))
        compare_img.paste(enhanced_img[0], (img_size, 0))

        # 保存对比图
        compare_img.save(output_paths[0])
        print(f"保存到: {output_paths[0]}")

def parse_args():
    parser = argparse.ArgumentParser(description="批量处理图片")
    parser.add_argument('--input_dir', type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument('--output_dir', type=str, required=True, help="输出文件夹路径")
    parser.add_argument('--batch_size', type=int, default=4, help="批量处理的图片数量")
    parser.add_argument('--img_size', type=int, default=1024, help="图像尺寸")
    parser.add_argument('--crop', action='store_true', help="是否进行裁剪")
    parser.add_argument('--prompt', type=str, default="a painting of a spaceship", help="用于生成的提示")
    parser.add_argument('--prompt_dict', type=str, help="包含图像和提示的字典")
    parser.add_argument('--short_side_pixel', type=int, default=2048, help="短边尺寸")
    parser.add_argument('--dit_path', type=str, help="DIT模型路径")
    parser.add_argument('--lora_path', type=str, help="LoRA模型路径")
    parser.add_argument('--enable_dit_fp8_computation', action='store_true', help="启用DIT FP8计算")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dit_path=args.dit_path,
        lora_path=args.lora_path,
        prompt=args.prompt,
        batch_size=args.batch_size,
        crop=args.crop,
        prompt_dict=args.prompt_dict,
        short_side_pixel=args.short_side_pixel,
        img_size=args.img_size,
        enable_dit_fp8_computation=args.enable_dit_fp8_computation
    )
