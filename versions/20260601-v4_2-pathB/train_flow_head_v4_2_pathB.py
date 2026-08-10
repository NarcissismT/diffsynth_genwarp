#!/usr/bin/env python3
"""
FlowHead V4.2 (Path B / DA-Flow 忠实版) 训练脚本
==================================================
核心修复 V4.1 的根因 bug：V4.1 的 Q/K 来自**同一张 warped 图**的同一组 token
（to_q 和 to_k 投影同一个 img_modulated），CorrBlock 里根本没有跨图位移信号
→ 这才是水波纹的根因，不是 grad loss / IN。

DA-Flow 的精髓是 "Q 来自 frame k，K 来自 frame k+1，且二者在同一个跨帧 attention 里
互相 attend 过"。我们核实了 Qwen-Image-Edit **原生就做跨图 attention**：
  diffsynth/pipelines/qwen_image.py:711-714  把 edit 图 token 拼进主序列
  diffsynth/models/qwen_image_dit.py:245-249 每个 block 内对 [主图;edit图] joint attention
  qwen_image.py:726  RoPE 用 img_shapes 双区域，两图自动拿不同位置偏移（不冲突）

所以 path B 不需要改 DiT、不需要 Stage-1 训练，只需在抽特征时：
  1. 把 corrected（main, 加噪）和 warped（edit, 干净）两图 token 拼接喂 DiT
  2. Q 从 corrected 段切，K 从 warped 段切（跨图！）
  3. 其余（DPT / CNN / CorrBlock / GRU / loss / EPE 日志）完全复用 V4.1

与 model_fn_qwen_image 的两处对齐（V4.1 都错了）：
  - timestep 必须 /1000（model_fn:706），V4.1 漏除
  - 两图拼接 + img_shapes 双区域 RoPE（V4.1 只有单图）

新增 --k_source 参数：
  warped    : K 从 warped 段切 —— 正式 path B（跨图）
  corrected : K 从 corrected 段切 —— sham 对照（Q/K 同图，等价 V4.1），
              用于 K-source 消融，判定"原生跨图注意力是否携带几何信号"。
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = "/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
_UPSTREAM = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio"
sys.path.insert(0, _HERE)               # 本目录下的 utils（self-contained）
sys.path.insert(0, _PROJECT_ROOT)
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

class QwenImageFlowV4_2TrainingModule(DiffusionTrainingModule):
    """
    V4.2 Path B：双图拼接 + 跨段 Q/K 提取。
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
        gradloss_ratio=1.0,
        dit_target_layers=None,
        dit_channels=3072,
        diff_out_ch=96,
        freeze_dit=True,
        loss_print_interval=100,
        k_source="warped",        # ← V4.2 新增：'warped'(正式) / 'corrected'(sham 对照)
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

        assert k_source in ("warped", "corrected"), f"k_source 只能是 warped/corrected，得到 {k_source}"
        self.k_source = k_source
        print(f"K 特征来源: {k_source}  "
              f"({'跨图(正式 path B)' if k_source == 'warped' else 'sham 对照(Q/K 同图)'})")

        # ---- FlowHead V4.1（结构完全复用）----
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
            print(f"FlowHead V4.2 加载初始权重: {flow_head_init}")
            print(f"  缺失 {len(missing)}，多余 {len(unexpected)}")
        else:
            print("FlowHead V4.2 随机初始化")

        for param in self.flow_head.parameters():
            param.requires_grad_(True)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.lambda_flow    = lambda_flow
        self.lambda_warp    = lambda_warp
        self.gradloss_ratio = gradloss_ratio

        self._step_counter = 0
        self._loss_print_interval = loss_print_interval

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable:,}（仅 FlowHead V4.2）")
        print(f"Loss config: lambda_flow={lambda_flow}, "
              f"lambda_warp={lambda_warp}, gradloss_ratio={gradloss_ratio}")

    # ------------------------------------------------------------------
    # forward_preprocess：main=corrected（加噪），edit=warped（干净）
    # ------------------------------------------------------------------
    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        # ★ V4.2 关键：main latent = corrected（GT 矫正图），edit_image = warped
        # 训练时 corrected 来自 data["image"]；推理时由 Qwen 50 步生成的 corrected_low 替代
        input_img = data["image"]
        inputs_shared = {
            "input_image": input_img,            # → input_latents（corrected）
            "edit_image":  data["edit_image"],   # → edit_latents（warped），走原生 EditImageEmbedder
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
    # 双图拼接 + 跨段 Q/K 提取（镜像 model_fn_qwen_image:704-733）
    # ------------------------------------------------------------------
    def _extract_qk_features(self, inputs: dict) -> tuple:
        with torch.no_grad():
            dit = self.pipe.dit.module if hasattr(self.pipe.dit, 'module') else self.pipe.dit

            # timestep：scheduler.timesteps[id] ∈ [0,1000)，model_fn 内部 /1000
            t_id = torch.randint(400, 600, (1,))
            timestep = self.pipe.scheduler.timesteps[t_id].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device)

            # main = corrected 加噪；edit = warped 干净（与 Qwen-Edit 推理一致：edit_image 不加噪）
            main_noisy = self.pipe.scheduler.add_noise(
                inputs["input_latents"], inputs["noise"], timestep)
            edit_lat = inputs["edit_latents"]

            h_val, w_val = inputs["height"], inputs["width"]

            # 两个图像区域的 shape（RoPE 用，决定两图位置不冲突）
            img_shapes = [
                (main_noisy.shape[0], main_noisy.shape[2] // 2, main_noisy.shape[3] // 2),
                (edit_lat.shape[0],   edit_lat.shape[2] // 2,   edit_lat.shape[3] // 2),
            ]
            prompt_emb      = inputs["prompt_emb"]
            prompt_emb_mask = inputs["prompt_emb_mask"]
            txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

            # main 图 token
            image = rearrange(main_noisy, "B C (H P) (W Q) -> B (H W) (C P Q)",
                              H=h_val // 16, W=w_val // 16, P=2, Q=2)
            seq_main = image.shape[1]   # corrected 段长度，用来切分 Q/K

            # edit 图 token（用 edit_lat 自身 H/W，对齐 model_fn:713）
            edit_tok = rearrange(edit_lat, "B C (H P) (W Q) -> B (H W) (C P Q)",
                                 H=edit_lat.shape[2] // 2, W=edit_lat.shape[3] // 2, P=2, Q=2)

            # 防御性断言：corrected 与 warped 的 token 数必须一致，否则跨段切 K 会错位，
            # 且 DPTHead reshape (B,T,C)->(B,C,h,w) 会抛晦涩的 RuntimeError。
            # 配对数据本应同尺寸；不一致时在这里早失败 + 清晰报错。
            assert edit_tok.shape[1] == seq_main, (
                f"corrected/warped token 数不一致: corrected={seq_main}, warped={edit_tok.shape[1]}。"
                f"main_noisy={tuple(main_noisy.shape)}, edit_lat={tuple(edit_lat.shape)}。"
                f"请确认 image 与 edit_image 同分辨率。"
            )

            # ★ 拼接：[corrected段 ; warped段]
            image = torch.cat([image, edit_tok], dim=1)
            image = dit.img_in(image)
            text  = dit.txt_in(dit.txt_norm(prompt_emb))
            # ★ timestep / 1000（对齐 model_fn:706），V4.1 漏了
            conditioning     = dit.time_text_embed(timestep / 1000, image.dtype)
            image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=main_noisy.device)

            q_features, k_features = [], []
            for block_idx, block in enumerate(dit.transformer_blocks):
                if block_idx in self.dit_target_layers:
                    img_normed = block.img_norm1(image)
                    img_mod_attn, _ = block.img_mod(conditioning).chunk(2, dim=-1)
                    img_modulated, _ = block._modulate(img_normed, img_mod_attn)
                    q_all = block.attn.to_q(img_modulated).float()
                    k_all = block.attn.to_k(img_modulated).float()
                    # ★ Q ← corrected 段（前 seq_main）
                    q_features.append(q_all[:, :seq_main])
                    # ★ K ← warped 段（后段）或 corrected 段（sham 对照）
                    if self.k_source == "warped":
                        k_features.append(k_all[:, seq_main:])
                    else:  # sham：K 也从 corrected 段切 → Q/K 同图（等价 V4.1）
                        k_features.append(k_all[:, :seq_main])

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

        # 1. 双图提取 Q/K
        q_features, k_features = self._extract_qk_features(inputs)

        # 2. PIL → tensor
        corrected_t = self.pipe.preprocess_image(corrected_img).to(
            device=device, dtype=torch.float32)
        warped_t = self.pipe.preprocess_image(warped_img).to(
            device=device, dtype=torch.float32)

        # 3. FlowHead V4.2 前向：fmap_c=(corrected,Q_corrected), fmap_w=(warped,K_warped)
        predictions = self.flow_head(corrected_t, warped_t,
                                     q_features, k_features, iters=4)

        # 4. flow_gt 缩放到预测分辨率
        flow_gt_t = (flow_gt.unsqueeze(0).to(device)
                     if flow_gt.dim() == 3 else flow_gt.to(device))
        _, _, fh, fw = predictions[-1].shape
        scale = float(fh) / flow_gt_t.shape[-2]
        flow_gt_r = F.interpolate(
            flow_gt_t, size=(fh, fw), mode="bilinear", align_corners=True) * scale

        # 5. Loss: L_flow + L_grad + lambda_warp * L_warp
        L_flow, L_grad = sequence_loss_with_grad(
            predictions, flow_gt_r,
            gamma=0.8, gradloss_ratio=self.gradloss_ratio,
        )

        # 6. L_warp
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

        total = (self.lambda_flow * L_flow + L_grad + self.lambda_warp * L_warp)

        # ---- Loss + EPE 日志（rank0）----
        self._step_counter += 1
        if self._step_counter % self._loss_print_interval == 0:
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            if rank == 0:
                with torch.no_grad():
                    pred_final = predictions[-1].detach()
                    epe_map = torch.norm(pred_final - flow_gt_r, dim=1)
                    epe     = epe_map.mean().item()
                    epe_1px = (epe_map < 1).float().mean().item()
                    epe_3px = (epe_map < 3).float().mean().item()
                    epe_5px = (epe_map < 5).float().mean().item()
                    # 平凡基线：全零流的 EPE（flow 真值的平均模长），用于判断模型有没有"学到东西"
                    zero_epe = torch.norm(flow_gt_r, dim=1).mean().item()
                    flow_max = pred_final.abs().max().item()
                    flow_std = pred_final.std().item()
                print(
                    f"\n[step {self._step_counter:>7d}][K={self.k_source}] "
                    f"L_flow={L_flow.item():.4f} L_grad={L_grad.item():.6f} "
                    f"L_warp={L_warp.item():.4f} total={total.item():.4f} "
                    f"| EPE={epe:.3f}px (zero={zero_epe:.3f}) "
                    f"<1px={epe_1px*100:.1f}% <3px={epe_3px*100:.1f}% <5px={epe_5px*100:.1f}% "
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
    parser.add_argument("--dit_target_layers", type=str, default="35",
                        help="DiT block 索引（0-based），逗号分隔。例：\"35\" 或 \"23,35,47\"。\n"
                             "层选择待优化（先用单层 35 验证水波纹修复）。\n"
                             "DA-Flow 论文用 {3,13,16,17}，但那是它 ~18 层小模型的选择；\n"
                             "Qwen-Image-Edit 有 60 层，按相对位置映射约为 {10,43,53,56}。\n"
                             "后续优化时建议用 zero-shot EPE probing 在 60 层里实测选层，"
                             "不要直接照搬论文数字。")
    parser.add_argument("--flow_head_init", type=str, default=None)
    parser.add_argument("--lambda_flow",    type=float, default=1.0)
    parser.add_argument("--lambda_warp",    type=float, default=0.5)
    parser.add_argument("--gradloss_ratio", type=float, default=1.0)
    parser.add_argument("--diff_out_ch",    type=int, default=96)
    parser.add_argument("--loss_print_interval", type=int, default=100)
    parser.add_argument("--k_source", type=str, default="warped",
                        choices=["warped", "corrected"],
                        help="K 特征来源：warped=跨图(正式 path B)；"
                             "corrected=sham 对照(Q/K 同图，等价 V4.1)，用于消融验证")
    # 注意：--lora_checkpoint 已由 qwen_image_parser() 提供
    args = parser.parse_args()

    target_layers = parse_layer_list(args.dit_target_layers)
    print(f"V4.2 Path B 训练，目标层: {target_layers}, "
          f"gradloss_ratio: {args.gradloss_ratio}, k_source: {args.k_source}")

    dataset = FlowDataset(args=args)

    model = QwenImageFlowV4_2TrainingModule(
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
        k_source=args.k_source,
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
