# V4.2 — Path B / DA-Flow 忠实版 — 2026-06-01

## 一句话

修复 V4.1 的**根因 bug**：V4.1 的 Q 和 K 来自**同一张 warped 图**，CorrBlock 里根本没有跨图位移信号 → 这才是水波纹的真因（不是 grad loss、不是 InstanceNorm）。V4.2 利用 Qwen-Image-Edit **原生的跨图 attention**，让 Q 来自 corrected、K 来自 warped，二者在同一个 attention 里互相 attend 过——这正是 DA-Flow 的 `Q_k / K_{k+1}` 在文档去扭曲场景的对应。

## 最关键的事实（决定了不用做 DA-Flow Stage-1）

逐行核实了真实代码：

- `diffsynth/pipelines/qwen_image.py:711-714` — 当 `edit_latents` 存在时，`image = torch.cat([主图token, edit图token], dim=1)`，**edit 图 token 拼进主序列**
- `diffsynth/models/qwen_image_dit.py:245-249` — 每个 block 内 `joint_q/k/v = cat([txt, img])` + 单次 attention，对 `[主图; edit图]` 做**联合 all-pairs 注意力**
- `qwen_image.py:726` — RoPE 用 `img_shapes`（双区域），两图自动拿不同位置偏移，**不冲突**

**结论：Qwen-Image-Edit 天生就有跨图 attention（这正是它做图像编辑的机制）。DA-Flow 第一阶段费力 "lift" 出来的能力，Qwen 本来就有。所以跳过 Stage-1，直接做 Stage-2（冻结 DiT + 训练 flow head）。**

## V4.2 vs V4.1 改了什么

只新建 1 个训练脚本，flow_head 完全复用。两处实质改动都在 `_extract_qk_features`：

| 项 | V4.1（错） | V4.2（对） |
|----|-----------|-----------|
| 喂进 DiT 的图 | 只有 warped 一张 | **corrected(main,加噪) + warped(edit,干净) 拼接** |
| Q 来源 | warped 的 `to_q(img_modulated)` | **corrected 段** `to_q(...)[:, :seq_main]` |
| K 来源 | warped 的 `to_k(img_modulated)`（同一张图！）| **warped 段** `to_k(...)[:, seq_main:]`（跨图）|
| timestep | `time_text_embed(timestep)`（漏除 1000）| **`time_text_embed(timestep/1000)`**（对齐 model_fn:706）|
| 跨图位移信号 | 无（Q/K 同图）→ 水波纹 | 有（Q/K 跨图）|

其余（DPTHeads / HybridEncoder / CorrBlock / ConvGRU / FlowPredHead / sequence_loss_with_grad / gradient_loss / EPE 日志）**verbatim 复用** V4.1。

`main=corrected / edit=warped` 的角色判定理由：与 FlowHeadV4_1 契约一致（`fmap_c=feat_encoder(corrected,F_Q)`、`fmap_w=feat_encoder(warped,F_K)`、`CorrBlock(fmap_c,fmap_w)`），flow 语义为 corrected→warped 采样网格，正好喂 `warp_batch(warped, flow) ≈ corrected`。

## 文件结构

```
20260601-v4_2-pathB/
├── README.md                              # 本文档
├── train_flow_head_v4_2_pathB.py          # 训练主脚本（含 --k_source 开关）
├── utils/
│   └── flow_head_v4_1_grad.py             # 复用 V4.1 模型（self-contained 副本）
└── scripts/
    ├── train_v4_2_pathB_layer36.sh        # 正式 path B：k_source=warped（跨图）
    └── train_v4_2_sham_layer36.sh         # sham 对照：k_source=corrected（同图，等价 V4.1）
```

DiT / pipeline **零改动**（原生跨图 attention + RoPE 双区域）。

## K-source 消融：最关键的 go/no-go 闸门

最大风险：Qwen 原生跨图 attention 在机械上存在，但其预训练权重在 corrected↔warped 之间可能只学了语义/文本对应、**几何对应近乎为零**。若如此，K_warped 段只携带 warped 外观、不含相对 corrected 的几何位移，CorrBlock 退化回 V4.1 水平。

**用一个开关 `--k_source` 在同一脚本里做对照，2-3K 步定生死：**

- `--k_source warped`（正式）：K 从 warped 段切 → 真·跨图
- `--k_source corrected`（sham）：K 也从 corrected 段切 → Q/K 同图，等价 V4.1

**判据：3K 步内，正式版的 EPE / <3px% 必须明显优于 sham。**
- 若有明显优势 → 跨图 attention 确实携带几何，path B 成立，继续训
- 若两者重合（ΔEPE < ~0.3px 且 <3px% 几乎一样）→ 原生跨图 attention 不携带几何，path B 前提失败，需升级到真正的 Stage-1 LoRA（训 to_q/to_k/...）或回退 RAFT

日志里新增了 `zero=` 字段（全零流的 EPE，即 flow 真值平均模长）——**模型 EPE 必须明显低于 zero，否则说明 head 啥也没学到。**

## 怎么跑

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp

# 正式 path B
bash versions/20260601-v4_2-pathB/scripts/train_v4_2_pathB_layer36.sh

# sham 对照（配对跑，判定跨图注意力是否携带几何）
bash versions/20260601-v4_2-pathB/scripts/train_v4_2_sham_layer36.sh
```

ckpt 输出：`/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_2_ckpts/`

## 训练配置

| 项 | 取值 |
|----|------|
| 可训练参数 | 仅 FlowHead V4.2（~2M），DiT 全冻结 |
| LR | 1e-4，ConstantLR |
| gradloss_ratio | 1.0（出现水波纹再升 2-3）|
| lambda_flow / lambda_warp | 1.0 / 0.5 |
| RAFT iters | 训练 4 |
| 分辨率 | 512×512（main 和 edit 都是，token 数各 1024）|

## 损失日志"work"时应显示

- `EPE` 持续下降且明显低于 `zero=`（全零流基线）
- `<1px% / <3px% / <5px%` 单调上升
- `pred_max / std` 不塌缩到 ≈0（塌缩 = head 学会忽略 correlation）
- **正式版 EPE 明显优于 sham 对照**（最关键）

## 诚实预期

Path B 是修复 V4.1 根因的正确改动，成本极低（无 Stage-1、~2M 参数）。但在原始 EPE 指标上**大概率打不过已用相同标签训成熟的预训练 RAFT 基线（dewarp_dino）**——后者推理还不用先跑 50 步 Qwen。它的价值在于：验证"Qwen-Edit 联合注意力是否携带可用几何"，以及在退化/低对比文档上的潜在鲁棒性。**先用 K-source 对照（2-3K 步）作为是否继续投入的硬闸门。**

## 推理（待做）

训练出 ckpt 后，推理需要：warped → Qwen 50 步生成 corrected_low → 把 (corrected_low, warped) 双图喂 DiT 提取 Q/K → FlowHead → warp 原图。推理脚本将基于 `versions/20260526-v4_1-grad-loss/qwen_image_flow_v4_1.py` 改造（patched_call 里同样改成双图拼接 + 跨段切 Q/K）。
```
注意训练-推理不对称：训练 Q 来自 corrected_GT，推理来自 Qwen 生成的 corrected_low（有 VAE 损耗）。
若验证 EPE 在 corrected_GT 与 corrected_low 输入间 gap > 30%，需在训练中混入预生成的 corrected_low。
```
