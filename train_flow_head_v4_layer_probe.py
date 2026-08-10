#!/usr/bin/env python3
"""
FlowHead V4 Layer Probing 训练脚本
====================================
目标：系统验证 Qwen Image DiT 哪些层的特征对文档几何矫正最有用。

与 V3 的核心区别：
  V3：--num_dit_layers N  → 固定取最后 N 层（block 60-N ~ block 59）
  V4：--dit_target_layers "11,23,35,47"  → 任意指定层（0-based index）

训练策略（与 V3 相同，Stage 2 冻结 DiT）：
  1. VAE encode warped → latent
  2. 用 warped latent 做一次 DiT 单步前向，在 --dit_target_layers 指定层提取 Q/K
  3. FlowHeadV4(corrected, warped, q_feats, k_feats) → flow_pred
  4. Loss = lambda_flow * L_flow + lambda_warp * L_warp

注意：V4 默认去掉 L_diffusion（DiT 完全冻结，不参与梯度），
      若需要联合训练可加 --no_freeze_dit 参数。

推荐实验脚本见 scripts/train_exp_A_layer12.sh ~ train_exp_F_layer12_24_36_48.sh
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)
sys.path.insert(0, _UPSTREAM)

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, ImageDataset, ModelLogger,
    launch_training_task, qwen_image_parser,
)
from utils.flow_head_v4_layer_probe import FlowHeadV4LayerProbe, sequence_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def warp_batch(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """批量 backward warping。img: (B,3,H,W)[-1,1], flow: (B,2,H,W) 像素偏移。"""
    B, _, H, W = flow.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = base + torch.stack([
        flow[:, 0] * 2.0 / (W - 1),
        flow[:, 1] * 2.0 / (H - 1),
    ], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# Dataset（与 V3 相同）
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
# 训练模块
# ---------------------------------------------------------------------------

class QwenImageFlowV4TrainingModule(DiffusionTrainingModule):
    """
    V4 训练模块：支持任意 DiT 层选择的光流预测训练。

    dit_target_layers: list[int]，0-based，要提取 Q/K 的 DiT block 索引
                       例如 [11] 对应 Layer 12；[11,23,35,47] 对应 Layer 12/24/36/48

    freeze_dit: True（默认）→ DiT 完全冻结，不计算 L_diffusion，显存更省
    """

    def __init__(
        self,
        model_paths=None,
        tokenizer_path=None,
        processor_path=None,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        flow_head_init=None,
        lambda_flow=1.0,
        lambda_warp=0.5,
        dit_target_layers=None,      # list[int]，0-based block 索引
        dit_channels=3072,
        diff_out_ch=96,
        freeze_dit=True,
    ):
        super().__init__()

        # ---- 加载 pipeline ----
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs = [ModelConfig(path=path) for path in model_paths]

        tokenizer_config = (ModelConfig(tokenizer_path) if tokenizer_path
                            else ModelConfig(model_id="Qwen/Qwen-Image",
                                            origin_file_pattern="tokenizer/"))
        processor_config = (ModelConfig(processor_path) if processor_path
                            else ModelConfig(model_id="Qwen/Qwen-Image-Edit",
                                            origin_file_pattern="processor/"))

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])

        # ---- 可选：加载已有 LoRA 权重（只用于特征提取，不继续训练 LoRA）----
        # 必须先 add_lora_to_model 再 load_state_dict，否则 LoRA key 找不到目标模块
        if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
            lora_target_modules = [
                "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
                "to_out.0", "to_add_out", "img_mlp.net.2", "img_mod.1",
                "txt_mlp.net.2", "txt_mod.1",
            ]
            dit_with_lora = self.add_lora_to_model(
                self.pipe.dit,
                target_modules=lora_target_modules,
                lora_rank=32,
            )
            sd = load_state_dict(lora_checkpoint)
            sd = self.mapping_lora_state_dict(sd)
            dit_with_lora.load_state_dict(sd, strict=False)
            self.pipe.dit = dit_with_lora
            print(f"加载 LoRA checkpoint: {lora_checkpoint}")

        # ---- 冻结 DiT ----
        for param in self.pipe.parameters():
            param.requires_grad_(False)
        if freeze_dit:
            print("DiT 完全冻结（不训练，不计算 L_diffusion）")

        # ---- 确定目标层 ----
        if dit_target_layers is None:
            dit_target_layers = [11, 23, 35, 47]  # 默认：Layer 12/24/36/48
        self.dit_target_layers = set(dit_target_layers)
        self.max_target_layer  = max(self.dit_target_layers)
        num_dit_layers = len(self.dit_target_layers)
        print(f"DiT 目标层（0-based）: {self.dit_target_layers}")

        # ---- FlowHead V4 ----
        self.flow_head = FlowHeadV4LayerProbe(
            iters=4,
            dit_channels=dit_channels,
            diff_out_ch=diff_out_ch,
            num_dit_layers=num_dit_layers,
        )
        if flow_head_init and os.path.exists(flow_head_init):
            self.flow_head.load_state_dict(
                torch.load(flow_head_init, map_location="cpu"))
            print(f"FlowHead V4 初始权重: {flow_head_init}")
        else:
            print("FlowHead V4 随机初始化")

        for param in self.flow_head.parameters():
            param.requires_grad_(True)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.lambda_flow  = lambda_flow
        self.lambda_warp  = lambda_warp

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable:,}（仅 FlowHead V4）")

    # ------------------------------------------------------------------
    # forward_preprocess：准备 prompt_emb / input_latents 等，供特征提取用
    # ------------------------------------------------------------------

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        # input_image 用 warped（edit_image），使 input_latents 来自弯曲图，
        # 与推理时一致（推理时只有 warped，没有 rectified）
        input_img = data.get("edit_image") or data["image"]
        inputs_shared = {
            "input_image": input_img,
            "height": input_img.size[1],
            "width":  input_img.size[0],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    # ------------------------------------------------------------------
    # 核心：在指定层提取 Q/K 特征
    # ------------------------------------------------------------------

    def _extract_qk_features(self, inputs: dict) -> tuple:
        """
        对 warped 图的 latent 做一次 DiT 单步前向，
        在 self.dit_target_layers 指定的每一层提取 image-stream Q/K 特征。

        提前在 max_target_layer 处退出，避免跑后续无关层（节省显存和时间）。

        Returns:
            q_features: list[L] of (B, T, C)，T = (H/16)*(W/16)
            k_features: list[L] of (B, T, C)
        """
        with torch.no_grad():
            dit_raw = self.pipe.dit.module if hasattr(self.pipe.dit, 'module') else self.pipe.dit

            # 使用随机 timestep（中等噪声，σ ≈ 0.4~0.6 对应训练 step ~400~600）
            timestep_id = torch.randint(400, 600, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device)

            latents = self.pipe.scheduler.add_noise(
                inputs["input_latents"], inputs["noise"], timestep)

            h_val = inputs["height"]
            w_val = inputs["width"]

            img_shapes = [(latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2)]
            prompt_emb      = inputs["prompt_emb"]
            prompt_emb_mask = inputs["prompt_emb_mask"]
            txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

            image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)",
                              H=h_val // 16, W=w_val // 16, P=2, Q=2)
            image = dit_raw.img_in(image)
            text  = dit_raw.txt_in(dit_raw.txt_norm(prompt_emb))
            conditioning     = dit_raw.time_text_embed(timestep, image.dtype)
            image_rotary_emb = dit_raw.pos_embed(img_shapes, txt_seq_lens,
                                                  device=latents.device)

            q_features, k_features = [], []
            for block_idx, block in enumerate(dit_raw.transformer_blocks):
                if block_idx in self.dit_target_layers:
                    # 提取该层 attention 前的 image Q/K（与 V3 逻辑相同）
                    img_normed = block.img_norm1(image)
                    img_mod_attn, _ = block.img_mod(conditioning).chunk(2, dim=-1)
                    img_modulated, _ = block._modulate(img_normed, img_mod_attn)
                    q_features.append(block.attn.to_q(img_modulated).float())
                    k_features.append(block.attn.to_k(img_modulated).float())

                text, image = block(
                    image=image, text=text, temb=conditioning,
                    image_rotary_emb=image_rotary_emb,
                )

                if block_idx == self.max_target_layer:
                    break  # 最深目标层跑完，剩余层不跑（省显存）

        return q_features, k_features

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        flow_gt    = data.get("flow_gt", None)
        corrected_img = data.get("image",      None)
        warped_img    = data.get("edit_image", None)

        if (flow_gt is None or corrected_img is None or warped_img is None
                or self.lambda_flow <= 0):
            # 没有 flow 标签时返回 0，不 crash
            dummy = next(self.flow_head.parameters())
            return torch.tensor(0.0, device=dummy.device, requires_grad=True)

        device = next(self.flow_head.parameters()).device

        # 1. 提取 DiT Q/K 特征
        q_features, k_features = self._extract_qk_features(inputs)

        # 2. PIL → tensor
        corrected_t = self.pipe.preprocess_image(corrected_img).to(
            device=device, dtype=torch.float32)
        warped_t = self.pipe.preprocess_image(warped_img).to(
            device=device, dtype=torch.float32)

        # 3. FlowHead V4 前向
        predictions = self.flow_head(corrected_t, warped_t,
                                     q_features, k_features, iters=4)

        # 4. flow_gt 缩放到预测分辨率
        flow_gt_t = (flow_gt.unsqueeze(0).to(device)
                     if flow_gt.dim() == 3 else flow_gt.to(device))
        _, _, fh, fw = predictions[-1].shape
        scale = float(fh) / flow_gt_t.shape[-2]
        flow_gt_r = F.interpolate(
            flow_gt_t, size=(fh, fw), mode="bilinear", align_corners=True) * scale

        # 5. L_flow（序列 L1 loss）
        L_flow = sequence_loss(predictions, flow_gt_r, gamma=0.8)

        # 6. L_warp（warp 重建 loss）
        if self.lambda_warp > 0:
            flow_final = predictions[-1]
            warp_result = warp_batch(
                F.interpolate(warped_t, size=(fh, fw), mode="bilinear", align_corners=True),
                flow_final,
            )
            correct_r = F.interpolate(corrected_t, size=(fh, fw),
                                       mode="bilinear", align_corners=True)
            L_warp = F.l1_loss(warp_result, correct_r)
        else:
            L_warp = torch.tensor(0.0, device=device)

        return self.lambda_flow * L_flow + self.lambda_warp * L_warp


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def parse_layer_list(s: str) -> list:
    """将 "11,23,35,47" 解析为 [11, 23, 35, 47]。"""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = qwen_image_parser()
    parser.add_argument(
        "--dit_target_layers", type=str, default="11,23,35,47",
        help=(
            "要提取 Q/K 的 DiT block 索引（0-based），逗号分隔。\n"
            "示例：\n"
            "  单层 Layer 12 → \"11\"\n"
            "  三层 Layer 24/36/48 → \"23,35,47\"\n"
            "  四层 Layer 12/24/36/48 → \"11,23,35,47\""
        ),
    )
    parser.add_argument("--flow_head_init", type=str, default=None,
                        help="FlowHead V4 初始权重，不传则随机初始化")
    parser.add_argument("--lambda_flow",    type=float, default=1.0)
    parser.add_argument("--lambda_warp",    type=float, default=0.5)
    parser.add_argument("--diff_out_ch",    type=int, default=96,
                        help="DPT Head 输出通道数")
    args = parser.parse_args()

    target_layers = parse_layer_list(args.dit_target_layers)
    print(f"Layer probing 实验，目标层（0-based）: {target_layers}")

    dataset = FlowDataset(args=args)

    model = QwenImageFlowV4TrainingModule(
        model_paths=args.model_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        flow_head_init=args.flow_head_init,
        lambda_flow=args.lambda_flow,
        lambda_warp=args.lambda_warp,
        dit_target_layers=target_layers,
        diff_out_ch=args.diff_out_ch,
        freeze_dit=True,
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
