#!/usr/bin/env python3
"""
FlowHead V4.4 (Path B + 难度加权 + 梯度累积) 训练脚本
======================================================
继承 V4.2 的全部架构（双图跨段 Q/K，不改任何网络结构），针对【过拟合诊断】
定位的真实瓶颈做两处训练侧改进：

诊断结论（high-only 长训，log 74932）：
  6 个 high 样本单独训 5000 步 → <5px 99.5%（含 168px 极端样本）。
  但 18 样本混训只到 53%。→ 不是架构/容量/搜索范围瓶颈，
  而是【难样本(大位移)欠拟合 + 训练未收敛】：high 收敛慢，混训时被 low/mid 占满步数。

V4.4 改进（均为训练侧，不破坏单模型约束）：
  1. 难度加权 loss：按样本 zero-flow EPE 加权（--hard_weight_alpha），
     难样本权重更大 → 更多有效梯度 → 解决 high 欠拟合。
  2. 梯度累积接线：V4.2 的 --gradient_accumulation_steps 是死参数（没传给
     launch_training_task），V4.4 接线 → 等效 batch 加大 → 降噪 + 加速收敛。

继承 V4.2 的正确逻辑：双图拼接、跨段 Q/K(Q←corrected/K←warped)、timestep/1000、
  grad loss、InstanceNorm、--k_source 开关。架构与 ckpt 与 V4.2 完全兼容。
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
from PIL import Image
from tqdm import tqdm

from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, ImageDataset, ModelLogger,
    launch_training_task, qwen_image_parser,
)
from utils.flow_head_v4_4 import (
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
    """
    V4.4 扩展：支持 VAE round-trip 域对齐。
    vae_roundtrip_prob > 0 时，按该概率把 main 图(data["image"]=corrected GT)
    替换成 corrected_vae（GT 经 VAE 编码-解码 round-trip 的版本），
    使训练 main 图更接近推理时 Qwen 生成的 corrected_low（也有 VAE 损耗）→ 缩小域差。
    只换 main 图，warped(edit_image) 和 flow_gt 保持不变。
    """
    def __init__(self, *args, vae_roundtrip_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_roundtrip_prob = vae_roundtrip_prob

    def __getitem__(self, data_id):
        import random as _random
        data = super().__getitem__(data_id)
        if data is None:
            return None
        meta = self.data[data_id % len(self.data)]
        # flow_gt
        flow_path = meta.get("flow_gt_path", "")
        if flow_path and os.path.exists(flow_path):
            data["flow_gt"] = torch.from_numpy(
                np.load(flow_path).astype(np.float32))
        else:
            data["flow_gt"] = None
        # VAE round-trip 域对齐：按概率把 main 图换成 corrected_vae
        if self.vae_roundtrip_prob > 0 and _random.random() < self.vae_roundtrip_prob:
            vae_path = meta.get("corrected_vae_path", "")
            if vae_path and os.path.exists(vae_path) and data.get("image") is not None:
                try:
                    # resize 到与已加载 main 图相同尺寸（保证与 flow_gt/warped 对齐）
                    target_w, target_h = data["image"].size  # PIL: (W, H)
                    vae_img = Image.open(vae_path).convert("RGB").resize(
                        (target_w, target_h), Image.BILINEAR)
                    data["image"] = vae_img
                    data["_used_vae"] = True
                except Exception:
                    pass
        return data


# ---------------------------------------------------------------------------
# 训练模块
# ---------------------------------------------------------------------------

class QwenImageFlowV4_4TrainingModule(DiffusionTrainingModule):
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
        hard_weight_alpha=0.0,    # ← V4.4 新增：难度加权指数。0=关闭(等价V4.2)
        hard_weight_ref=24.0,     # ← V4.4 新增：zero-flow EPE 归一化参考(数据集中位数≈24px)
        hard_weight_max=3.0,      # ← V4.4 新增：权重上限(防极端样本爆炸)
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
        self.hard_weight_alpha = hard_weight_alpha   # V4.4
        self.hard_weight_ref   = hard_weight_ref
        self.hard_weight_max   = hard_weight_max
        self._w_running        = 0.0   # raw_w 的 running 均值（用于归一化）

        self._step_counter = 0
        self._loss_print_interval = loss_print_interval

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable:,}（仅 FlowHead V4.4）")
        print(f"Loss config: lambda_flow={lambda_flow}, "
              f"lambda_warp={lambda_warp}, gradloss_ratio={gradloss_ratio}")
        print(f"难度加权: alpha={hard_weight_alpha} "
              f"({'启用' if hard_weight_alpha > 0 else '关闭(等价 V4.2)'}), "
              f"ref={hard_weight_ref}, max={hard_weight_max}")

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
    def _extract_qk_features(self, inputs: dict, fixed_t_id: int = None) -> tuple:
        with torch.no_grad():
            dit = self.pipe.dit.module if hasattr(self.pipe.dit, 'module') else self.pipe.dit

            # timestep：scheduler.timesteps[id] ∈ [0,1000)，model_fn 内部 /1000
            # 训练：随机 [400,600) 增强；验证：固定 fixed_t_id（与推理 500 对齐、可复现）
            if fixed_t_id is not None:
                t_id = torch.tensor([int(fixed_t_id)])
            else:
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

        # ---- V4.4 难度加权（修复版）：按样本 zero-flow EPE 重新分配梯度 ----
        # 修复 v1 的两个 bug：
        #   bug1: 之前用 flow_gt_r(H/8 + 数值缩放 0.0625) 的 EPE 算 w，但 ref=24 是
        #         原图尺度的中位数 → 尺度不匹配 → w 全 <0.3，难样本没被突出，反而整体降权。
        #   bug2: w 不做均值归一化 → 整体 loss 尺度漂移，拖慢收敛。
        # 修复：(a) 用【原图尺度】的 zero_epe（缩放前的 flow_gt_t）；
        #      (b) 用 running 均值把 w 归一化到均值≈1 —— 只改"相对难度分配"，
        #          不改整体 loss 尺度（难样本 w>1 升权、简单 w<1 降权，平均仍≈1）。
        if self.hard_weight_alpha > 0:
            with torch.no_grad():
                # 原图尺度 zero-flow EPE（用缩放前的 flow_gt_t）
                zero_epe_orig = torch.norm(flow_gt_t, dim=1).mean().item()
                raw_w = (max(zero_epe_orig, 1e-3) / self.hard_weight_ref) ** self.hard_weight_alpha
                # running 均值归一化（EMA），让权重围绕 1 分布，不改整体 loss 尺度
                self._w_running = (0.99 * self._w_running + 0.01 * raw_w
                                   if self._w_running > 0 else raw_w)
                w = raw_w / max(self._w_running, 1e-3)
                # clamp 防极端样本（168px）爆炸，下界防过度抑制简单样本
                w = float(min(max(w, 1.0 / self.hard_weight_max), self.hard_weight_max))
            total = total * w
        else:
            w = 1.0

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
                    f"L_warp={L_warp.item():.4f} w={w:.2f} total={total.item():.4f} "
                    f"| EPE={epe:.3f}px (zero={zero_epe:.3f}) "
                    f"<1px={epe_1px*100:.1f}% <3px={epe_3px*100:.1f}% <5px={epe_5px*100:.1f}% "
                    f"| pred_max={flow_max:.2f} std={flow_std:.3f}",
                    flush=True,
                )

        return total

    # ------------------------------------------------------------------
    # 验证：单样本 EPE（按推理那种 —— main 图用预生成的真扩散 corrected_low）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_one(self, corrected_low_pil, warped_pil, flow_gt_t, prompt,
                 fixed_t_id: int = 500, noise_seed: int = 0):
        """
        对一个验证样本算 EPE/<Npx。与训练 forward 同链路（forward_preprocess +
        _extract_qk_features + flow_head），唯三区别：
          1. main 图 = 预生成的真扩散 corrected_low（暴露域差），不是 GT corrected
          2. timestep 固定（默认 500，对齐推理）+ noise 固定 seed → 可复现、跨 ckpt 可比
          3. 不算 loss/不反传/不做难度加权
        返回 dict: epe / <1px / <3px / <5px / zero_epe（均为 float）。
        """
        device = next(self.flow_head.parameters()).device
        # 复用 forward_preprocess：main=corrected_low, edit=warped
        data = {"image": corrected_low_pil, "edit_image": warped_pil, "prompt": prompt}
        inputs = self.forward_preprocess(data)
        # 固定 noise，保证验证可复现（forward_preprocess 内部会用 inputs["noise"]）。
        # generator 设备必须与 noise 张量设备一致。
        if inputs.get("noise", None) is not None:
            ndev = inputs["noise"].device
            gen = torch.Generator(device=ndev).manual_seed(int(noise_seed))
            inputs["noise"] = torch.randn(
                inputs["noise"].shape, generator=gen,
                device=ndev, dtype=inputs["noise"].dtype)

        q_features, k_features = self._extract_qk_features(inputs, fixed_t_id=fixed_t_id)

        corrected_t = self.pipe.preprocess_image(corrected_low_pil).to(
            device=device, dtype=torch.float32)
        warped_t = self.pipe.preprocess_image(warped_pil).to(
            device=device, dtype=torch.float32)

        was_training = self.flow_head.training
        self.flow_head.eval()
        pred = self.flow_head(corrected_t, warped_t, q_features, k_features, iters=4)
        if isinstance(pred, list):
            pred = pred[-1]
        if was_training:
            self.flow_head.train()

        flow_gt_t = (flow_gt_t.unsqueeze(0) if flow_gt_t.dim() == 3 else flow_gt_t).to(device)
        _, _, fh, fw = pred.shape
        scale = float(fh) / flow_gt_t.shape[-2]
        flow_gt_r = F.interpolate(flow_gt_t, size=(fh, fw),
                                  mode="bilinear", align_corners=True) * scale
        epe_map = torch.norm(pred - flow_gt_r, dim=1)
        return {
            "epe":      epe_map.mean().item(),
            "lt1px":    (epe_map < 1).float().mean().item(),
            "lt3px":    (epe_map < 3).float().mean().item(),
            "lt5px":    (epe_map < 5).float().mean().item(),
            "zero_epe": torch.norm(flow_gt_r, dim=1).mean().item(),
        }


# ---------------------------------------------------------------------------
# 验证器（held-out 集，main 图用预生成的真扩散 corrected_low）
# ---------------------------------------------------------------------------

class FlowValidator:
    """
    持有 held-out 验证集（gen_val_corrected_low.py 产出的 val_index.csv），
    在训练循环中周期性调用 model.eval_one() 算 EPE/<5px，并分 low/mid/high 报告。

    关键：main 图读【预生成的真扩散 corrected_low】，真实暴露训练-推理域差。
    验证指标与训练日志同流打印，并追加到 val_metrics.csv，供对比过拟合。
    """
    def __init__(self, val_index_csv: str, default_prompt: str,
                 metrics_csv: str = None, max_samples: int = 0):
        import pandas as pd
        self.df = pd.read_csv(val_index_csv)
        if max_samples > 0:
            self.df = self.df.iloc[:max_samples].copy()
        self.default_prompt = default_prompt
        self.metrics_csv = metrics_csv
        self._header_written = False
        print(f"[Validator] 加载验证集 {len(self.df)} 行 ← {val_index_csv}")

    def _load_flow_gt(self, path):
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        return torch.from_numpy(np.load(path).astype(np.float32))

    @torch.no_grad()
    def run(self, model, step: int):
        """跑完整验证集，返回汇总 dict 并打印/落盘。只应在 rank0 调用。"""
        # DDP 下取真正的 module
        m = model.module if hasattr(model, "module") else model
        per_diff = {"low": [], "mid": [], "high": []}
        all_epe, all_lt5, all_lt3, all_lt1, all_zero = [], [], [], [], []

        for _, row in self.df.iterrows():
            flow_gt = self._load_flow_gt(row["flow_gt_path"])
            if flow_gt is None:
                continue
            try:
                corrected_low = Image.open(row["corrected_low_path"]).convert("RGB")
                warped = Image.open(row["edit_image"]).convert("RGB").resize(
                    corrected_low.size, Image.BILINEAR)
                prompt = row.get("prompt", self.default_prompt)
                if not isinstance(prompt, str) or not prompt.strip():
                    prompt = self.default_prompt
                r = m.eval_one(corrected_low, warped, flow_gt, prompt)
            except Exception as e:
                print(f"[Validator] 样本失败 {row.get('corrected_low_path','?')}: {e}")
                continue
            all_epe.append(r["epe"]); all_lt5.append(r["lt5px"])
            all_lt3.append(r["lt3px"]); all_lt1.append(r["lt1px"])
            all_zero.append(r["zero_epe"])
            diff = str(row.get("difficulty", "")).strip()
            if diff in per_diff:
                per_diff[diff].append(r["lt5px"])

        if not all_epe:
            print(f"[VAL step {step}] 无有效样本")
            return None

        import numpy as _np
        summ = {
            "step": step,
            "val_epe":  float(_np.mean(all_epe)),
            "val_zero": float(_np.mean(all_zero)),
            "val_lt1":  float(_np.mean(all_lt1)) * 100,
            "val_lt3":  float(_np.mean(all_lt3)) * 100,
            "val_lt5":  float(_np.mean(all_lt5)) * 100,
            "n": len(all_epe),
        }
        for d in ("low", "mid", "high"):
            summ[f"val_lt5_{d}"] = (float(_np.mean(per_diff[d])) * 100
                                    if per_diff[d] else -1.0)

        print(
            f"\n[VAL step {step:>7d}] (真扩散 corrected_low, n={summ['n']}) "
            f"EPE={summ['val_epe']:.3f}px (zero={summ['val_zero']:.3f}) "
            f"<1px={summ['val_lt1']:.1f}% <3px={summ['val_lt3']:.1f}% <5px={summ['val_lt5']:.1f}% "
            f"| 分层<5px: low={summ['val_lt5_low']:.1f}% "
            f"mid={summ['val_lt5_mid']:.1f}% high={summ['val_lt5_high']:.1f}%",
            flush=True,
        )

        if self.metrics_csv:
            import csv
            write_header = not os.path.exists(self.metrics_csv)
            with open(self.metrics_csv, "a", newline="") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=list(summ.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(summ)
        return summ


def launch_training_task_with_val(
    dataset, model, model_logger, optimizer, scheduler,
    num_workers=8, save_steps=None, num_epochs=1,
    gradient_accumulation_steps=1, find_unused_parameters=False,
    validator: "FlowValidator" = None, val_interval: int = 4000,
):
    """
    本地副本：等同 diffsynth.trainers.utils.launch_training_task，额外加验证钩子。
    不修改共享框架文件。验证只在 rank0 跑，跑完 barrier 同步，model 训练态不变。
    """
    from accelerate import Accelerator, DistributedDataParallelKwargs
    import torch.distributed as dist

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler)

    # step_counter 与 ckpt 文件名 / 训练日志 [step N] 同口径：每个 data 迭代(micro-step)都 +1。
    # 注意：on_step_end 的 num_steps 也是每 micro-step +1，所以 val_interval 直接和 save_steps 同单位可比。
    # 验证只在【梯度累积边界 sync_gradients】触发，避免在累积窗口中途翻转 train/eval。
    # 故建议 val_interval 为 gradient_accumulation_steps 的整数倍（默认 4000 % 2 == 0 ✓）。
    step_counter = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

            step_counter += 1
            if (validator is not None and val_interval > 0
                    and accelerator.sync_gradients
                    and step_counter % val_interval == 0):
                if accelerator.is_main_process:
                    was_training = accelerator.unwrap_model(model).flow_head.training
                    validator.run(model, step_counter)
                    if was_training:
                        accelerator.unwrap_model(model).flow_head.train()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


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
    # V4.4 新增：难度加权（解决 high 样本欠拟合）
    parser.add_argument("--hard_weight_alpha", type=float, default=1.0,
                        help="难度加权指数。0=关闭(等价V4.2)；1.0=线性按位移加权(推荐)；"
                             ">1 更激进。w=clamp((zero_epe/ref)^alpha, max)")
    parser.add_argument("--hard_weight_ref", type=float, default=24.0,
                        help="zero-flow EPE 归一化参考（数据集中位数≈24px）")
    parser.add_argument("--hard_weight_max", type=float, default=3.0,
                        help="权重上限（防极端样本 168px 爆炸）")
    # V4.4 新增：VAE round-trip 域对齐
    parser.add_argument("--vae_roundtrip_prob", type=float, default=0.5,
                        help="训练时把 main 图(corrected GT)按此概率换成 corrected_vae"
                             "(GT 经 VAE round-trip)，缩小与推理 corrected_low 的域差。"
                             "0=关闭(纯 GT)；0.5=一半样本用 vae 版(推荐)。"
                             "需 CSV 含 corrected_vae_path 列。")
    # V4.4 新增：held-out 验证集（main 图用预生成的真扩散 corrected_low）
    parser.add_argument("--val_index_csv", type=str, default=None,
                        help="gen_val_corrected_low.py 产出的 val_index.csv。"
                             "提供时开启训练内嵌验证（按推理那种：真扩散 corrected_low）。"
                             "不提供则等同原训练（无验证）。")
    parser.add_argument("--val_interval", type=int, default=4000,
                        help="每多少个【优化步】跑一次验证（建议与 save_steps 对齐）")
    parser.add_argument("--val_max_samples", type=int, default=0,
                        help=">0 时只验证前 N 个样本（调试用）")
    parser.add_argument("--val_metrics_csv", type=str, default=None,
                        help="验证指标追加写到此 CSV；默认写到 output_path/val_metrics.csv")
    # 注意：--lora_checkpoint / --gradient_accumulation_steps 由 qwen_image_parser() 提供
    args = parser.parse_args()

    target_layers = parse_layer_list(args.dit_target_layers)
    print(f"V4.4 训练，目标层: {target_layers}, gradloss_ratio: {args.gradloss_ratio}, "
          f"k_source: {args.k_source}, hard_weight_alpha: {args.hard_weight_alpha}, "
          f"grad_accum: {args.gradient_accumulation_steps}, "
          f"vae_roundtrip_prob: {args.vae_roundtrip_prob}")

    dataset = FlowDataset(args=args, vae_roundtrip_prob=args.vae_roundtrip_prob)

    model = QwenImageFlowV4_4TrainingModule(
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
        hard_weight_alpha=args.hard_weight_alpha,
        hard_weight_ref=args.hard_weight_ref,
        hard_weight_max=args.hard_weight_max,
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

    # 验证集：提供 --val_index_csv 则走带验证钩子的本地训练循环（不改共享框架）
    if args.val_index_csv:
        default_prompt = ("Flatten this warped or curled document image to a flat, "
                          "undistorted version. Preserve all text, lines, and content "
                          "accurately.")
        metrics_csv = args.val_metrics_csv or os.path.join(
            args.output_path, "val_metrics.csv")
        os.makedirs(args.output_path, exist_ok=True)
        validator = FlowValidator(
            val_index_csv=args.val_index_csv,
            default_prompt=default_prompt,
            metrics_csv=metrics_csv,
            max_samples=args.val_max_samples,
        )
        print(f"内嵌验证已开启：每 {args.val_interval} 优化步验证一次 → {metrics_csv}")
        launch_training_task_with_val(
            dataset, model, model_logger, optimizer, scheduler,
            num_epochs=args.num_epochs,
            save_steps=args.save_steps,
            find_unused_parameters=args.find_unused_parameters,
            num_workers=args.dataset_num_workers,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            validator=validator,
            val_interval=args.val_interval,
        )
    else:
        launch_training_task(
            dataset, model, model_logger, optimizer, scheduler,
            num_epochs=args.num_epochs,
            save_steps=args.save_steps,
            find_unused_parameters=args.find_unused_parameters,
            num_workers=args.dataset_num_workers,
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # V4.4 修复：接线
        )


if __name__ == "__main__":
    main()
