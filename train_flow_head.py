#!/usr/bin/env python3
"""
FlowHead V2 + LoRA 联合训练脚本
================================
在 QwenImage DiT LoRA 微调的基础上，同时训练 FlowHead V2。

联合 Loss：
  L = L_diffusion + lambda_flow * L_flow + lambda_warp * L_warp

  L_diffusion  扩散模型标准 MSE loss（与 train_sample.sh 完全一致）
  L_flow       序列 L1 loss，FlowHeadV2(corrected_gt, warped) vs flow_gt
  L_warp       L1(grid_sample(warped, flow_pred), corrected_gt)

训练目标：
  - LoRA 权重：让 DiT 学习几何矫正
  - FlowHead V2 权重：从 (corrected_gt, warped) 像素图预测光流

用法见 scripts/train_flow_head.sh
"""

import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)
sys.path.insert(0, _UPSTREAM)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, ImageDataset, ModelLogger,
    launch_training_task, qwen_image_parser,
)
from utils.flow_head_v2 import FlowHeadV2, sequence_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def pil_list_to_tensor(imgs, device, dtype=torch.float32):
    """PIL 列表 → (B,3,H,W)，值域 [-1,1]，与 pipe.preprocess_image 等价。"""
    arrs = [np.array(img, dtype=np.float32) * (2.0 / 255.0) - 1.0 for img in imgs]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(device=device, dtype=dtype)


def warp_batch(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """批量 backward warping。img: (B,3,H,W)[-1,1], flow: (B,2,H,W) 像素。"""
    B, _, H, W = flow.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = base + torch.stack([flow[:, 0] * 2.0 / (W - 1),
                                flow[:, 1] * 2.0 / (H - 1)], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# 扩展 Dataset：额外加载 flow_gt
# ---------------------------------------------------------------------------

class FlowDataset(ImageDataset):
    """
    在 ImageDataset 基础上，额外从 flow_gt_path 列加载光流 numpy。
    跳过 flow_gt_path 不存在的行（允许部分数据没有光流标签）。
    """

    def __getitem__(self, data_id):
        data = super().__getitem__(data_id)
        if data is None:
            return None
        # 加载 flow_gt（如果有）
        flow_path = self.data[data_id % len(self.data)].get("flow_gt_path", "")
        if flow_path and os.path.exists(flow_path):
            data["flow_gt"] = torch.from_numpy(
                np.load(flow_path).astype(np.float32))  # (2, H, W)
        else:
            data["flow_gt"] = None
        return data


# ---------------------------------------------------------------------------
# 联合训练模块
# ---------------------------------------------------------------------------

class QwenImageFlowTrainingModule(DiffusionTrainingModule):
    """
    继承自 DiffusionTrainingModule，在扩散 loss 上叠加 flow loss。
    可训练参数：LoRA 权重 + FlowHead 权重。
    """

    def __init__(
        self,
        model_paths=None,
        tokenizer_path=None,
        processor_path=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        flow_head_init=None,
        lambda_flow=1.0,
        lambda_warp=0.1,
    ):
        super().__init__()

        # ---- 加载 pipeline ----
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs = [ModelConfig(path=path) for path in model_paths]

        tokenizer_config = ModelConfig(tokenizer_path) if tokenizer_path else ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
        processor_config = ModelConfig(processor_path) if processor_path else ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )

        # 重置训练 scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # ---- 冻结不训练的模块 ----
        self.pipe.freeze_except([])

        # ---- 添加 LoRA ----
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint: {lora_checkpoint}")
            setattr(self.pipe, lora_base_model, model)

        # ---- 挂载 FlowHead V2（独立模块，不依赖 latent）----
        self.flow_head_v2 = FlowHeadV2(iters=4)
        if flow_head_init and os.path.exists(flow_head_init):
            self.flow_head_v2.load_state_dict(
                torch.load(flow_head_init, map_location="cpu"))
            print(f"FlowHead V2 初始权重: {flow_head_init}")
        else:
            print("FlowHead V2 随机初始化")

        # FlowHead V2 参数设为可训练
        for param in self.flow_head_v2.parameters():
            param.requires_grad_(True)

        # ---- 其他配置 ----
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.lambda_flow = lambda_flow
        self.lambda_warp = lambda_warp

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable:,}（LoRA + FlowHead V2）")

    def forward_preprocess(self, data):
        """与原 QwenImageTrainingModule 完全一致。"""
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        controlnet_input, blockwise_controlnet_input = {}, {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
            elif extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        if len(blockwise_controlnet_input) > 0:
            inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(**blockwise_controlnet_input)]
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        # ---- L_diffusion：标准扩散 MSE loss ----
        L_diffusion = self.pipe.training_loss(**models, **inputs)

        # ---- L_flow + L_warp（仅当 flow_gt 存在时计算）----
        flow_gt = data.get("flow_gt", None)
        if flow_gt is not None and self.lambda_flow > 0:
            device = next(self.flow_head_v2.parameters()).device

            corrected_img = data.get("image", None)      # PIL，GT 矫正图
            warped_img    = data.get("edit_image", None)  # PIL，弯曲输入图

            if corrected_img is not None and warped_img is not None:
                # PIL → tensor [-1, 1]，(1, 3, H, W)
                corrected_t = self.pipe.preprocess_image(corrected_img).to(device=device, dtype=torch.float32)
                warped_t    = self.pipe.preprocess_image(warped_img).to(device=device, dtype=torch.float32)

                # FlowHead V2 前向（训练模式返回 predictions list）
                predictions = self.flow_head_v2(corrected_t, warped_t, iters=4)

                # flow_gt 缩放到与预测相同的分辨率
                flow_gt_t = flow_gt.unsqueeze(0).to(device) if flow_gt.dim() == 3 else flow_gt.to(device)
                _, _, fh, fw = predictions[-1].shape
                scale = float(fh) / flow_gt_t.shape[-2]
                flow_gt_resized = F.interpolate(
                    flow_gt_t, size=(fh, fw), mode="bilinear", align_corners=True) * scale

                # 序列 L1 loss
                L_flow = sequence_loss(predictions, flow_gt_resized, gamma=0.8)

                # L_warp（用最终预测光流 warp warped 图，对比 corrected_gt）
                if self.lambda_warp > 0:
                    flow_final = predictions[-1]
                    warp_result = warp_batch(
                        F.interpolate(warped_t,    size=(fh, fw), mode="bilinear", align_corners=True),
                        flow_final,
                    )
                    correct_resized = F.interpolate(corrected_t, size=(fh, fw), mode="bilinear", align_corners=True)
                    L_warp = F.l1_loss(warp_result, correct_resized)
                else:
                    L_warp = torch.tensor(0.0, device=device)

                loss = L_diffusion + self.lambda_flow * L_flow + self.lambda_warp * L_warp
            else:
                loss = L_diffusion
        else:
            loss = L_diffusion

        return loss


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    # 复用 qwen_image_parser，追加 FlowHead 专属参数
    parser = qwen_image_parser()
    parser.add_argument("--flow_head_init",   type=str, default=None,
                        help="FlowHead 初始权重（.pth），不传则随机初始化")
    parser.add_argument("--lambda_flow",      type=float, default=1.0,
                        help="L_flow 权重，0 表示不计算")
    parser.add_argument("--lambda_warp",      type=float, default=0.1,
                        help="L_warp 权重")
    args = parser.parse_args()

    dataset = FlowDataset(args=args)

    model = QwenImageFlowTrainingModule(
        model_paths=args.model_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=getattr(args, "lora_checkpoint", None),
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        flow_head_init=args.flow_head_init,
        lambda_flow=args.lambda_flow,
        lambda_warp=args.lambda_warp,
    )

    # 保存 LoRA 权重 + FlowHead 权重
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
    )


if __name__ == "__main__":
    main()
