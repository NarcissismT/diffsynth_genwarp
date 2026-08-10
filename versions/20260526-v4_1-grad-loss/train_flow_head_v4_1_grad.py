#!/usr/bin/env python3
"""
FlowHead V4.1 训练脚本 — 加 gradient loss + InstanceNorm
=========================================================
相对 V4 (train_flow_head_v4_layer_probe.py) 的改动：

1. 模型用 FlowHeadV4_1（utils/flow_head_v4_1_grad.py）
   - ContextEncoder 内 BatchNorm2d → InstanceNorm2d，无 BN buffer 问题
2. Loss 改用 sequence_loss_with_grad
   - 每次 RAFT 迭代的 flow 都加 L2 梯度惩罚
   - Total = L_flow + L_grad + lambda_warp * L_warp
3. 新增 --gradloss_ratio 参数（默认 1.0，参考 dewarp_dino）

其他流程（pipeline units 准备 inputs / DiT 单步特征提取 / freeze_dit）
与 V4 完全一致，只改 loss 和模型 ContextEncoder。
"""

import os
import sys
import json

# 项目根目录（用于 import utils 等）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = "/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)               # 本目录下的 utils
sys.path.insert(0, _PROJECT_ROOT)       # 主项目（备用）
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
# 注意：从本地 utils（versions/20260526-v4_1-grad-loss/utils）导入
from utils.flow_head_v4_1_grad import (
    FlowHeadV4_1,
    sequence_loss_with_grad,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def warp_batch(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
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
# Dataset
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

class QwenImageFlowV4_1TrainingModule(DiffusionTrainingModule):
    """
    V4.1 训练模块：在 V4 基础上加 gradient loss + IN 替代 BN
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
        gradloss_ratio=1.0,                # ← V4.1 新增
        dit_target_layers=None,
        dit_channels=3072,
        diff_out_ch=96,
        freeze_dit=True,
        loss_print_interval=100,
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

        # ---- 加载 LoRA（用于特征提取，不更新）----
        if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
            lora_target_modules = [
                "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
                "to_out.0", "to_add_out", "img_mlp.net.2", "img_mod.1",
                "txt_mlp.net.2", "txt_mod.1",
            ]
            dit_with_lora = self.add_lora_to_model(
                self.pipe.dit, target_modules=lora_target_modules, lora_rank=32)
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
            dit_target_layers = [11, 23, 35, 47]
        self.dit_target_layers = set(dit_target_layers)
        self.max_target_layer  = max(self.dit_target_layers)
        num_dit_layers = len(self.dit_target_layers)
        print(f"DiT 目标层（0-based）: {sorted(self.dit_target_layers)}")

        # ---- FlowHead V4.1 ----
        self.flow_head = FlowHeadV4_1(
            iters=4,
            dit_channels=dit_channels,
            diff_out_ch=diff_out_ch,
            num_dit_layers=num_dit_layers,
        )
        if flow_head_init and os.path.exists(flow_head_init):
            from safetensors.torch import load_file
            init_sd = load_file(flow_head_init) if flow_head_init.endswith(".safetensors") \
                      else torch.load(flow_head_init, map_location="cpu")
            missing, unexpected = self.flow_head.load_state_dict(init_sd, strict=False)
            print(f"FlowHead V4.1 加载初始权重: {flow_head_init}")
            print(f"  缺失 {len(missing)}（含 BN→IN 转换会导致部分缺失，正常），"
                  f"多余 {len(unexpected)}")
        else:
            print("FlowHead V4.1 随机初始化")

        for param in self.flow_head.parameters():
            param.requires_grad_(True)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.lambda_flow    = lambda_flow
        self.lambda_warp    = lambda_warp
        self.gradloss_ratio = gradloss_ratio   # V4.1 新增

        # Loss 打印计数器（launch_training_task 不打印 loss，自己打印）
        self._step_counter = 0
        self._loss_print_interval = loss_print_interval

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable:,}（仅 FlowHead V4.1）")
        print(f"Loss config: lambda_flow={lambda_flow}, "
              f"lambda_warp={lambda_warp}, gradloss_ratio={gradloss_ratio}")

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        # input_image = warped (与 V4 一致)
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

    def _extract_qk_features(self, inputs: dict) -> tuple:
        with torch.no_grad():
            dit_raw = self.pipe.dit.module if hasattr(self.pipe.dit, 'module') else self.pipe.dit

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
            image_rotary_emb = dit_raw.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

            q_features, k_features = [], []
            for block_idx, block in enumerate(dit_raw.transformer_blocks):
                if block_idx in self.dit_target_layers:
                    img_normed = block.img_norm1(image)
                    img_mod_attn, _ = block.img_mod(conditioning).chunk(2, dim=-1)
                    img_modulated, _ = block._modulate(img_normed, img_mod_attn)
                    q_features.append(block.attn.to_q(img_modulated).float())
                    k_features.append(block.attn.to_k(img_modulated).float())

                text, image = block(image=image, text=text, temb=conditioning,
                                     image_rotary_emb=image_rotary_emb)

                if block_idx == self.max_target_layer:
                    break

        return q_features, k_features

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        flow_gt    = data.get("flow_gt", None)
        corrected_img = data.get("image",      None)
        warped_img    = data.get("edit_image", None)

        if (flow_gt is None or corrected_img is None or warped_img is None
                or self.lambda_flow <= 0):
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

        # 3. FlowHead V4.1 前向（4 次 RAFT 迭代）
        predictions = self.flow_head(corrected_t, warped_t,
                                     q_features, k_features, iters=4)

        # 4. flow_gt 缩放到预测分辨率
        flow_gt_t = (flow_gt.unsqueeze(0).to(device)
                     if flow_gt.dim() == 3 else flow_gt.to(device))
        _, _, fh, fw = predictions[-1].shape
        scale = float(fh) / flow_gt_t.shape[-2]
        flow_gt_r = F.interpolate(
            flow_gt_t, size=(fh, fw), mode="bilinear", align_corners=True) * scale

        # 5. V4.1 核心 Loss: L_flow + L_grad + lambda_warp * L_warp
        L_flow, L_grad = sequence_loss_with_grad(
            predictions, flow_gt_r,
            gamma=0.8, gradloss_ratio=self.gradloss_ratio,
        )

        # 6. L_warp（warp 重建 loss，与 V4 相同）
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

        total = (self.lambda_flow * L_flow
               + L_grad                            # gradloss_ratio 已在内部乘过
               + self.lambda_warp * L_warp)

        # ---- Loss 打印（每 N 步一次，方便观察各项量级）----
        # 注意：8 卡 DDP 每个 rank 都会跑 forward，每张卡各自计数。
        # 只在 rank0 打印（其他 rank 打印会被覆盖在 tqdm 进度条里看不到）
        self._step_counter += 1
        if self._step_counter % self._loss_print_interval == 0:
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            if rank == 0:
                # ---- EPE 监控指标（不参与反向，仅显示）----
                # EPE = sqrt((dx_pred-dx_gt)² + (dy_pred-dy_gt)²) per pixel
                # 然后取均值。这是 RAFT/FlowNet 的标准评估指标
                with torch.no_grad():
                    pred_final = predictions[-1].detach()
                    epe_map = torch.norm(pred_final - flow_gt_r, dim=1)  # (B, H, W)
                    epe       = epe_map.mean().item()
                    epe_1px   = (epe_map < 1).float().mean().item()
                    epe_3px   = (epe_map < 3).float().mean().item()
                    epe_5px   = (epe_map < 5).float().mean().item()
                    flow_max  = pred_final.abs().max().item()
                    flow_std  = pred_final.std().item()

                # 用 \n 强制换行，避免被 tqdm 进度条覆盖
                print(
                    f"\n[step {self._step_counter:>7d}] "
                    f"L_flow={L_flow.item():.4f} "
                    f"L_grad={L_grad.item():.6f} "
                    f"L_warp={L_warp.item():.4f} "
                    f"total={total.item():.4f} "
                    f"| EPE={epe:.3f}px "
                    f"<1px={epe_1px*100:.1f}% "
                    f"<3px={epe_3px*100:.1f}% "
                    f"<5px={epe_5px*100:.1f}% "
                    f"| pred_max={flow_max:.2f} std={flow_std:.3f}",
                    flush=True,
                )

        return total


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def parse_layer_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = qwen_image_parser()
    parser.add_argument(
        "--dit_target_layers", type=str, default="35",
        help="DiT block 索引（0-based），逗号分隔。例：\"35\" 或 \"23,35,47\"",
    )
    parser.add_argument("--flow_head_init", type=str, default=None,
                        help="FlowHead 初始权重（可加载 V4 ckpt 续训）")
    parser.add_argument("--lambda_flow",    type=float, default=1.0)
    parser.add_argument("--lambda_warp",    type=float, default=0.5)
    parser.add_argument("--gradloss_ratio", type=float, default=1.0,
                        help="V4.1 新增：gradient loss 权重，参考 dewarp_dino。"
                             "推荐 1.0，太弱压不住水波纹，太强 (>5) 会过平滑")
    parser.add_argument("--diff_out_ch",    type=int, default=96)
    parser.add_argument("--loss_print_interval", type=int, default=100,
                        help="每多少步打印一次 loss（rank0 only），默认 100")
    # 注意：--lora_checkpoint 已由 qwen_image_parser() 提供，这里不要再加，否则 argparse 报 conflict
    args = parser.parse_args()

    target_layers = parse_layer_list(args.dit_target_layers)
    print(f"V4.1 训练，目标层: {target_layers}, gradloss_ratio: {args.gradloss_ratio}")

    dataset = FlowDataset(args=args)

    model = QwenImageFlowV4_1TrainingModule(
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
        gradloss_ratio=args.gradloss_ratio,
        dit_target_layers=target_layers,
        diff_out_ch=args.diff_out_ch,
        freeze_dit=True,
        loss_print_interval=args.loss_print_interval,
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
