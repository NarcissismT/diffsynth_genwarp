#!/usr/bin/env python3
"""
FlowHead V3 (DA-Flow 风格) 联合训练脚本
========================================
在 QwenImage DiT LoRA 微调的基础上，同时训练 FlowHead V3。

V3 vs V2 的差异：
  V2：FlowHeadV2(corrected, warped) → flow
  V3：FlowHeadV3DAFlow(corrected, warped, q_features, k_features) → flow
      q_features / k_features 来自 DiT 推理时 top-L 层的 Q/K 注意力特征

联合 Loss（与 V2 完全相同）：
  L = L_diffusion + lambda_flow * L_flow + lambda_warp * L_warp

训练策略（参考 DA-Flow 两阶段）：
  Stage 1（已有）：DiT + LoRA 微调（train_flow_head.sh）
  Stage 2（本脚本）：冻结 DiT，训练 DPT Head + FlowHead V3 的 CNN/GRU 部分

用法见 scripts/train_flow_head_v3_daflow.sh
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
from utils.flow_head_v3_daflow import FlowHeadV3DAFlow, sequence_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# 辅助函数（与 train_flow_head.py 相同）
# ---------------------------------------------------------------------------

def warp_batch(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
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
# Dataset（与 train_flow_head.py 相同）
# ---------------------------------------------------------------------------

class FlowDataset(ImageDataset):
    def __getitem__(self, data_id):
        data = super().__getitem__(data_id)
        if data is None:
            return None
        flow_path = self.data[data_id % len(self.data)].get("flow_gt_path", "")
        if flow_path and os.path.exists(flow_path):
            data["flow_gt"] = torch.from_numpy(
                np.load(flow_path).astype(np.float32))
        else:
            data["flow_gt"] = None
        return data


# ---------------------------------------------------------------------------
# 联合训练模块（V3 DA-Flow 风格）
# ---------------------------------------------------------------------------

class QwenImageFlowV3TrainingModule(DiffusionTrainingModule):
    """
    V3 版本：FlowHead V3 使用 DiT Q/K 特征。

    训练时：
      1. VAE encode corrected_gt → input_latents（用于扩散 loss）
      2. 用 input_latents 做一次 DiT 前向（去噪到 t=0），提取 Q/K 特征
         （简化：直接在训练的去噪 step 里 hook，不额外跑完整推理）
      3. FlowHead V3(corrected_gt, warped, q_feats, k_feats) → flow_pred
      4. Loss = L_diffusion + lambda_flow * L_flow + lambda_warp * L_warp

    注意：训练时提取 Q/K 特征的方法是在 training_loss 的单步前向里 hook，
    而不是跑完整的 50 步推理（太慢）。这和 DA-Flow 的两阶段训练一致：
    Stage 2 冻结 DiT，训练时只做单步前向提取特征。
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
        # V3 专属
        dit_channels=3072,
        diff_out_ch=96,
        num_dit_layers=4,
        freeze_dit=True,   # Stage 2：冻结 DiT，只训练 DPT Head + FlowHead
    ):
        super().__init__()

        # ---- 加载 pipeline ----
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs = [ModelConfig(path=path) for path in model_paths]

        tokenizer_config = ModelConfig(tokenizer_path) if tokenizer_path else \
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
        processor_config = ModelConfig(processor_path) if processor_path else \
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])

        # ---- LoRA ----
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
            )
            if lora_checkpoint is not None:
                sd = load_state_dict(lora_checkpoint)
                sd = self.mapping_lora_state_dict(sd)
                model.load_state_dict(sd, strict=False)
                print(f"LoRA checkpoint: {lora_checkpoint}")
            setattr(self.pipe, lora_base_model, model)

        # ---- FlowHead V3 ----
        self.flow_head_v3 = FlowHeadV3DAFlow(
            iters=4,
            dit_channels=dit_channels,
            diff_out_ch=diff_out_ch,
            num_dit_layers=num_dit_layers,
        )
        if flow_head_init and os.path.exists(flow_head_init):
            self.flow_head_v3.load_state_dict(
                torch.load(flow_head_init, map_location="cpu"))
            print(f"FlowHead V3 初始权重: {flow_head_init}")
        else:
            print("FlowHead V3 随机初始化")

        # ---- 冻结策略 ----
        if freeze_dit:
            # Stage 2：冻结 DiT（含 LoRA），只训练 FlowHead V3
            for param in self.pipe.parameters():
                param.requires_grad_(False)
            for param in self.flow_head_v3.parameters():
                param.requires_grad_(True)
            print("Stage 2 模式：冻结 DiT + LoRA，只训练 FlowHead V3")
        else:
            # Stage 1+2 联合：LoRA + FlowHead V3 都训练
            for param in self.flow_head_v3.parameters():
                param.requires_grad_(True)
            print("联合训练模式：LoRA + FlowHead V3")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.lambda_flow = lambda_flow
        self.lambda_warp = lambda_warp
        self.num_dit_layers = num_dit_layers

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable:,}")

    def forward_preprocess(self, data):
        """与 train_flow_head.py 完全一致。"""
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
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                pass
            elif extra_input.startswith("controlnet_"):
                pass
            else:
                inputs_shared[extra_input] = data[extra_input]
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def _extract_dit_features(self, inputs: dict) -> tuple:
        """
        在 DiT 的单步前向里 hook Q/K 特征。
        使用随机 timestep（与 training_loss 逻辑一致），
        提取 top-num_dit_layers 层的 img Q/K。

        Returns:
            q_features: list[L] of (B, T, C)
            k_features: list[L] of (B, T, C)
        """
        dit = self.pipe.dit
        num_blocks = len(dit.transformer_blocks)
        target_indices = list(range(num_blocks - self.num_dit_layers, num_blocks))

        # 完全不用 hook，直接自己跑 DiT 的部分前向
        # 只跑到 target_indices 的最后一层，在每个目标 block 处手动提取 Q/K
        with torch.no_grad():
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device)
            noisy = self.pipe.scheduler.add_noise(
                inputs["input_latents"], inputs["noise"], timestep)

            # 拿到原始 dit（unwrap DDP）
            dit_raw = dit.module if hasattr(dit, 'module') else dit

            # 重现 DiT forward 的前置处理
            latents = noisy
            height = inputs.get("height", self.pipe.device)
            width  = inputs.get("width",  self.pipe.device)
            # 从 inputs 里获取 height/width
            h_val = inputs["height"] if "height" in inputs else latents.shape[2] * 8
            w_val = inputs["width"]  if "width"  in inputs else latents.shape[3] * 8

            img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
            prompt_emb      = inputs["prompt_emb"]
            prompt_emb_mask = inputs["prompt_emb_mask"]
            txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

            from einops import rearrange as _rearrange
            image = _rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)",
                               H=h_val//16, W=w_val//16, P=2, Q=2)
            image = dit_raw.img_in(image)
            text  = dit_raw.txt_in(dit_raw.txt_norm(prompt_emb))
            conditioning    = dit_raw.time_text_embed(timestep, image.dtype)
            image_rotary_emb = dit_raw.pos_embed(img_shapes, txt_seq_lens,
                                                  device=latents.device)

            q_features, k_features = [], []
            for block_idx, block in enumerate(dit_raw.transformer_blocks):
                # 在目标层，先提取 Q/K，再走 block forward
                if block_idx in target_indices:
                    img_normed   = block.img_norm1(image)
                    img_mod_attn, _ = block.img_mod(conditioning).chunk(2, dim=-1)
                    img_modulated, _ = block._modulate(img_normed, img_mod_attn)
                    q_features.append(block.attn.to_q(img_modulated).float())
                    k_features.append(block.attn.to_k(img_modulated).float())

                text, image = block(
                    image=image, text=text, temb=conditioning,
                    image_rotary_emb=image_rotary_emb,
                )

                if block_idx == max(target_indices):
                    break  # 最后一个目标层跑完就停，不需要跑后面的层

        return q_features, k_features

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        models = {name: getattr(self.pipe, name)
                  for name in self.pipe.in_iteration_models}

        # L_diffusion（只在非 freeze_dit 模式下有意义）
        L_diffusion = self.pipe.training_loss(**models, **inputs)

        # L_flow + L_warp
        flow_gt = data.get("flow_gt", None)
        if flow_gt is not None and self.lambda_flow > 0:
            device = next(self.flow_head_v3.parameters()).device

            corrected_img = data.get("image", None)
            warped_img    = data.get("edit_image", None)

            if corrected_img is not None and warped_img is not None:
                # 提取 DiT Q/K 特征（单步前向）
                q_features, k_features = self._extract_dit_features(inputs)

                # PIL → tensor
                corrected_t = self.pipe.preprocess_image(corrected_img).to(
                    device=device, dtype=torch.float32)
                warped_t = self.pipe.preprocess_image(warped_img).to(
                    device=device, dtype=torch.float32)

                # FlowHead V3 前向
                predictions = self.flow_head_v3(
                    corrected_t, warped_t, q_features, k_features, iters=4)

                # flow_gt 缩放
                flow_gt_t = flow_gt.unsqueeze(0).to(device) \
                    if flow_gt.dim() == 3 else flow_gt.to(device)
                _, _, fh, fw = predictions[-1].shape
                scale = float(fh) / flow_gt_t.shape[-2]
                flow_gt_r = F.interpolate(
                    flow_gt_t, size=(fh, fw), mode="bilinear",
                    align_corners=True) * scale

                L_flow = sequence_loss(predictions, flow_gt_r, gamma=0.8)

                if self.lambda_warp > 0:
                    flow_final = predictions[-1]
                    warp_result = warp_batch(
                        F.interpolate(warped_t, size=(fh, fw),
                                      mode="bilinear", align_corners=True),
                        flow_final,
                    )
                    correct_r = F.interpolate(corrected_t, size=(fh, fw),
                                              mode="bilinear", align_corners=True)
                    L_warp = F.l1_loss(warp_result, correct_r)
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
    parser = qwen_image_parser()
    parser.add_argument("--flow_head_init", type=str, default=None)
    parser.add_argument("--lambda_flow",    type=float, default=1.0)
    parser.add_argument("--lambda_warp",    type=float, default=0.1)
    parser.add_argument("--freeze_dit",     action="store_true", default=False,
                        help="Stage 2 模式：冻结 DiT，只训练 FlowHead V3")
    parser.add_argument("--diff_out_ch",    type=int, default=96)
    parser.add_argument("--num_dit_layers", type=int, default=4)
    args = parser.parse_args()

    dataset = FlowDataset(args=args)

    model = QwenImageFlowV3TrainingModule(
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
        freeze_dit=args.freeze_dit,
        diff_out_ch=args.diff_out_ch,
        num_dit_layers=args.num_dit_layers,
    )

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
