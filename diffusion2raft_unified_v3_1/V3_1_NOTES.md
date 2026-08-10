# v3.1 修改说明与训练检查表

## 这次解决的具体问题

v2 的推理结果虽然全局 EPE 下降，但表格横线和竖线仍有平滑弯曲。原因是白底像素在
全局 EPE 中占比很大，而 fold rate 只能发现翻折，无法发现没有翻折的低频弯线。

v3.1 不改模型参数结构，新增以下监督：

1. `structure_flow`：在文字边缘和长横/竖线上提高完整 flow 的监督权重。
2. `line_reconstruction`：重点比较重建图和 GT target 的线条像素。
3. `flow_gradient`：匹配预测 flow 与 GT flow 的一阶变化，而非把 flow 强行变平。
4. `curvature`：约束 `(pred_flow - GT_flow)` 的二阶变化，直接压制错误曲率。
5. `line_straightness`：对横线惩罚 y-error 沿 x 的波动，对竖线做对称约束。
6. `bending`：unified 阶段仅平滑受限 residual，权重从 `0.02` 降到 `0.002`，避免
   对完整 GT 所需曲率产生欠矫正偏置。

同时增加 `epe_p95`、`edge_epe`、`line_epe`、`line_normal_mae` 和
`line_straightness_error`，验证时按 `line_epe` 保存 `best.pt`。

## resize / flow 坐标安全规则

- `flow.shape[-2:]` 是 target sampling grid。
- `flow_source_size=[H,W]` 是 flow 数值对应的源图像素坐标画布。
- 当 flow grid 与 target 图像尺寸不同时，manifest 必须显式提供
  `flow_source_size`；不再把 flow grid 尺寸静默当成 source canvas。
- resize 先插值 absolute coordinate map，再分别变换 source x/y 坐标。

训练前执行：

```bash
python scripts/inspect_flow_sample.py \
  --manifest data/train.jsonl --index 0 \
  --output-dir tmp/flow_check_0 --max-mae 0.08
```

## 从 v2 epoch 8 续训

将权重放到 `runs/d2r/unified/resume_from_v2_epoch8.pt`，然后执行：

```bash
bash scripts/smoke_unified_v31.sh
# 查看 runs/preflight_v31/smoke/unified/previews/epoch_0009.jpg 后再继续
bash scripts/train_unified_v3.sh
```

模型参数和 optimizer group 不变，因此 model 权重与 Adam moments 可以恢复。新输出目录
为 `runs/d2r_v3_1/unified/`；推理优先使用 `best.pt`，不要默认使用 `latest.pt`。

`smoke_unified_v31.sh` 不是 lite 单元测试：它用 production 配置加载真实 Qwen，先执行
checkpoint/manifest/flow canvas 预检，再只跑一个训练 batch 和一个验证 batch。正式脚本也
默认执行预检，报告在 `runs/preflight_v31/main/preflight_report.json`；验证四联图则按 epoch
写入 `runs/d2r_v3_1/unified/previews/`，用于尽早发现 EPE 看不出的表格线波纹。

调试训练入口提供 `--max-train-steps`、`--max-val-batches`、`--output-dir` 和
`--preview-every`，因此短测试与正式 checkpoint 不再共用目录。

## 首轮训练应看什么

- 健康趋势：`line_epe`、`line_bend`、`epe_p95` 下降，且全局 EPE 不明显恶化。
- 安全条件：`fold_rate` 仍接近 0，`residual_p95` 不持续贴近 24 px 上限。
- Qwen 是否有用：`q_adv > 0`、`q_win > 0.5`，gate 逐步接近 `gate_target`。
- 如果 `line_bend` 降而 EPE 升很多，先把 `line_straightness` 从 0.10 降到 0.05。
- 如果 EPE 降而线仍弯，可把 `line_straightness` 从 0.10 升到 0.15；不要先增大
  absolute bending。

最终必须用固定真实表格集比较 prior、latest、best 三者，并同时记录 OCR CER、
`line_epe`、`line_bend` 和可视化结果。
