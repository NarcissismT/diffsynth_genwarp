# Qwen-Image-Edit 到 DocGrid-Flow v2 的数据流

最后更新：2026-07-31。规范版逐层说明和文件索引以
[`ARCHITECTURE_INDEX.md`](ARCHITECTURE_INDEX.md) 为准；本文保留原文件名，防止旧链接失效。

## 输入到底是什么

- 推理时模型只有一张输入：扭曲 RGB 图像 `Iw`。
- 训练时 `Mgt`、`valid_mask` 和矫正 RGB 只用于构造监督与 loss，不送入 Qwen/CNN 图像编码器。
- Qwen 只读取 `Iw`、固定几何 prompt、固定或受控 seed；它看不到 GT flow 或 GT 矫正图。

因此不是“双图 RAFT”，也不是“扭曲图 + 矫正图 + 光流一起输入”。

## 当前完整流程

```text
Iw
  |-- CNN/FPN -> local4 + fpn8
  |-- H/V Encoder -> hv4 + hv8 + H/V/boundary logits
  `-- Qwen VAE/condition preparation -> frozen Qwen DiT
       -> selected Hidden 或 Q/K -> DPT/FPN Adapter
                         |
          Qwen/CNN/HV 三路空间 softmax gated fusion
                         |
          Coarse Head -> Mc；Confidence Head -> C
                         |
          residual Coordinate Flow Transformer
          [Rt, Mc-P, P, C] tokenizer
          -> coordinate self-attention
          -> visual cross-attention
          -> H/V-gated FFN
          -> 2-channel velocity
                         |
          4-6 步 residual ODE -> Rhat
          Md = Mc + G(C) * Rhat
                         |
          map-aware resize 到 1/4
                         |
          WARR ConvGRU x4
          （warped CNN/HV + 一/二阶/mixed/Jacobian cues）
                         |
          learned update gate + bounded Delta-M
                         |
          RAFT-style convex map upsampling
                         |
          全分辨率 absolute backward map Mfinal
                         |
          grid_sample(原始高清 Iw, Mfinal)，仅一次
                         |
          最终矫正 RGB
```

## “解码器”分别是什么

1. `Qwen VAE Decoder`：不使用；Qwen pipeline 请求 latent，特征 hook 完成后丢弃 latent。
2. `Coordinate Flow Transformer`：几何主解码器，输出 residual velocity，不输出 RGB。
3. `WARR`：1/4 分辨率循环几何精修器，不直接读取 confidence。
4. `ConvexMapUpsampler`：WARR 后的最终 map 上采样层，替代 bilinear flow 上采样。
5. 最终 RGB“解码”只有对原始高清扭曲图的一次 `grid_sample`，不会经生成式 VAE 重画文字。

## Stage 5 patch 为什么仍是单图输入

`structure_patch` 只指定矩形输出画布中的 `target_window=[x0,y0,w,h]`。模型仍输入完整的扭曲
页面 `Iw`，CNN/HV/Qwen 都先看完整 source，再把条件特征采样到该 target window。对应 GT map
和预测 map 的数值始终是完整扭曲图中的 source `(x,y)`，不会减去 patch 左上角。因此它不是
“扭曲 patch + 矫正 patch”的双图输入，也不会造成训练 patch 与整页推理的坐标语义漂移。

## 训练阶段名

```text
coarse -> warr -> coord_fm -> qwen -> full_page
```

旧名字 `refiner`、`joint` 只作为兼容别名，分别映射到 `warr`、`full_page`。各阶段 Slurm
入口在 `slurm/docgrid_v2/`，每个阶段必须由 verified-GT Gate receipt 解锁下一阶段。

Stage 1–3 的配置显式设置 `instantiate_qwen_adapter=false`：它们不调用 Qwen pipeline，checkpoint
也不会混入未训练的 adapter 参数。Stage 4 才创建并训练 DPT/FPN adapter；它允许父 Stage-3
checkpoint 缺少这一组参数，并把 adapter 的重新初始化记录在自己的 config/checkpoint 中。Stage 5
则继承已经训练的 Stage-4 adapter。旧探索 checkpoint 中多出的未训练 adapter state 只会在当前
adapter-free 阶段被丢弃，其他任何 state mismatch 仍然 fail closed。

## 真实 Qwen 运行时验证

正式 Stage 4 前必须执行：

```bash
export DOCGRID_PYTHON=/path/to/qwen-env/bin/python
export DOCGRID_QWEN_PROBE_IMAGE=/path/to/warped_page.png
bash slurm/docgrid_v2/00_validate_qwen.sh --partition=a100
```

该检查使用真实本地 Qwen 权重，验证 block 的 `(text,image)` hidden 输出、目标/源图 token 切分、
选定 hidden 空间形状和有限值，并把 `vae.decode` 临时替换为禁止调用函数。正式 receipt v2 固定并
绑定 `512x512 / hidden / [-24,-12,-1] / bfloat16 / one probe step / seed 0 / CPU offload / latent
output`、3072 通道、每层实际 target/source shape，以及本地模型配置 SHA。Stage 4/5 对这些字段
逐项复核；用不同层号、dtype、尺寸、offload 设置或 lite/mock 产生的报告都会被拒绝。
