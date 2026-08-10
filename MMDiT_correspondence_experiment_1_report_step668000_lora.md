# Qwen-Image-Edit MMDiT LoRA 微调模型冻结对应评估实验一报告

## 单一结论

**条件性存在**。

本次使用 full（8+64+256）预注册规模。

最终排名只使用独立的 Confirmation 子集。最佳配置为 block `42`、
denoising step `24`、`post-RoPE`、
temperature `0.01`。

## 核心指标

| 指标 | 数值 |
|---|---:|
| Confirmation R@10, r=1 | 0.8321 |
| hard/extreme R@10, r=1 | 0.7631 |
| >96px 位移 R@10, r=1 | N/A |
| 文字/边缘 R@10, r=1 | 0.8402 |
| batch-shuffle R@10, r=1 | 0.1492 |
| random-candidate R@10, r=1 | 0.0215 |
| 相对最强对照倍数 | 5.578 |
| soft median EPE | 14.0216 px |
| identity median EPE | 9.6646 px |
| 相对 identity EPE 降幅 | -0.4508 |
| false identity rate | 0.0888 |
| 三 seed R@10 标准差 | 0.0113 |

## Confirmation 候选排名

| 排名 | 配置 | R@10,r=1 | soft median EPE (px) | false identity rate |
|---:|---|---:|---:|---:|
| 1 | L42/S24/post/tau=0.01 | 0.8321 | 14.0216 | 0.0888 |
| 2 | L42/S19/post/tau=0.01 | 0.8317 | 15.0369 | 0.0917 |
| 3 | L42/S29/post/tau=0.01 | 0.8265 | 13.5320 | 0.0859 |
| 4 | L42/S19/pre/tau=0.01 | 0.8211 | 15.6283 | 0.0922 |
| 5 | L37/S34/post/tau=0.01 | 0.7143 | 22.2066 | 0.0635 |
| 6 | L18/S44/pre/tau=0.01 | 0.1515 | 159.6586 | 0.0067 |

## pre-RoPE / post-RoPE

这里只比较由 Discovery 冻结后进入 Confirmation 的候选；完整全层对比见 Discovery 热力图。

| RoPE 状态 | 最佳入选配置 | R@10,r=1 | soft median EPE (px) |
|---|---|---:|---:|
| pre | L42/S19 | 0.8211 | 15.6283 |
| post | L42/S24 | 0.8321 | 14.0216 |

## Seed 稳定性

| 配置 | R@10 均值 | R@10 标准差 | soft median EPE 均值 | top-1 坐标方差 (token²) |
|---|---:|---:|---:|---:|
| L42/S19/post | 0.8381 | 0.0099 | 14.8746 | 25.5473 |
| L42/S24/post | 0.8342 | 0.0113 | 14.1802 | 22.7629 |
| L42/S29/post | 0.8289 | 0.0078 | 13.6236 | 22.9560 |

## 预注册强能力条件

- overall_recall: PASS
- hard_extreme_recall: PASS
- text_edge_recall: PASS
- control_multiple: PASS
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
| blank_or_low_texture | recall_at_10_r1=0.8296 |
| identity_bias_on_large_motion | false_identity_rate=0.0888 |
| seed_sensitivity | recall_standard_deviation=0.0113 |

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
