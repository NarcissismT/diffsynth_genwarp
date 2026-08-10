# Qwen-Image-Edit MMDiT 零样本文档对应评估实验一报告

## 单一结论

**条件性存在**。

本次使用 full（8+64+256）预注册规模。

最终排名只使用独立的 Confirmation 子集。最佳配置为 block `42`、
denoising step `9`、`post-RoPE`、
temperature `0.01`。

## 核心指标

| 指标 | 数值 |
|---|---:|
| Confirmation R@10, r=1 | 0.3338 |
| hard/extreme R@10, r=1 | 0.3217 |
| >96px 位移 R@10, r=1 | N/A |
| 文字/边缘 R@10, r=1 | 0.3730 |
| batch-shuffle R@10, r=1 | 0.1015 |
| random-candidate R@10, r=1 | 0.0215 |
| 相对最强对照倍数 | 3.290 |
| soft median EPE | 59.1322 px |
| identity median EPE | 9.6646 px |
| 相对 identity EPE 降幅 | -5.1184 |
| false identity rate | 0.0361 |
| 三 seed R@10 标准差 | 0.0082 |

## Confirmation 候选排名

| 排名 | 配置 | R@10,r=1 | soft median EPE (px) | false identity rate |
|---:|---|---:|---:|---:|
| 1 | L42/S09/post/tau=0.01 | 0.3338 | 59.1322 | 0.0361 |
| 2 | L42/S09/pre/tau=0.01 | 0.3325 | 61.5095 | 0.0367 |
| 3 | L42/S04/pre/tau=0.01 | 0.3090 | 54.8032 | 0.0294 |
| 4 | L46/S04/pre/tau=0.01 | 0.2975 | 83.4082 | 0.0348 |
| 5 | L37/S24/post/tau=0.01 | 0.2307 | 85.2087 | 0.0198 |
| 6 | L18/S39/pre/tau=0.01 | 0.0724 | 168.1453 | 0.0052 |

## pre-RoPE / post-RoPE

这里只比较由 Discovery 冻结后进入 Confirmation 的候选；完整全层对比见 Discovery 热力图。

| RoPE 状态 | 最佳入选配置 | R@10,r=1 | soft median EPE (px) |
|---|---|---:|---:|
| pre | L42/S09 | 0.3325 | 61.5095 |
| post | L42/S09 | 0.3338 | 59.1322 |

## Seed 稳定性

| 配置 | R@10 均值 | R@10 标准差 | soft median EPE 均值 | top-1 坐标方差 (token²) |
|---|---:|---:|---:|---:|
| L42/S04/pre | 0.3025 | 0.0098 | 55.6765 | 135.1323 |
| L42/S09/post | 0.3167 | 0.0082 | 58.8704 | 150.2427 |
| L42/S09/pre | 0.3177 | 0.0083 | 61.2072 | 169.8897 |

## 预注册强能力条件

- overall_recall: FAIL
- hard_extreme_recall: FAIL
- text_edge_recall: FAIL
- control_multiple: FAIL
- epe_reduction: FAIL
- false_identity: PASS
- seed_stability: PASS

## 预注册弱/不存在条件

- overall_recall: FAIL
- control_multiple: FAIL
- epe_not_better_than_identity: PASS
- hard_or_large_motion_near_chance: FAIL
- post_rope_identity_bias: FAIL
- seed_instability: FAIL

## 失败类型分解

| 类型 | 观测量 |
|---|---|
| large_displacement_gt_96px | recall_at_10_r1=N/A |
| blank_or_low_texture | recall_at_10_r1=0.3216 |
| identity_bias_on_large_motion | false_identity_rate=0.0361 |
| seed_sensitivity | recall_standard_deviation=0.0082 |

## 产物

- Discovery 热力图：`discovery/heatmaps/`
- Confirmation 总体指标：`confirmation/aggregate_metrics.json`
- Confirmation 分组指标：`confirmation/subgroup_metrics.json`
- Seed 稳定性：`seed_stability/seed_stability.json`
- 定性可视化：`visualizations/`
- 冻结配置与环境：`frozen_config.yaml`、`environment.json`

## 运行指纹

- model: `/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit`
- revision: `None`
- pipeline: `QwenImageEditPipeline`
- Diffusers: `0.35.2`
- prompt SHA-256: `7ca0327714e16bf1df20d65fffea957b02405bc64c84c8ee717ed93dbf9a7afd`
- scheduler config SHA-256: `e38ac5627ab954b5f5d0cb314a1cebcab5012ea12f5d398e7f2935bb081fe8e6`
- qualitative samples: `24`

注：聚合 JSON 中的 rate/mean 按有效 token 加权；EPE median/P95 使用可跨样本、
跨 rank 合并的等质量 centroid 估计 token-micro pooled quantile，并另存样本级 macro
均值。模型始终以 `output_type=latent` 运行，报告没有使用生成 RGB 做选择。
