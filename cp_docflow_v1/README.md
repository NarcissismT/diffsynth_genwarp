# DocGrid-Flow v2

这是 `Diffusion2RAFT_Plan_and_Goals_newest.md` 新范式的独立实现目录。旧的
`diffusion2raft_unified_v3_1/`、checkpoint 和 C4 diagnostic 保持冻结，不与这里共享
checkpoint schema 或坐标约定。

最新版的输入、逐层数据流、残差公式和规范文件名见
`docs/ARCHITECTURE_INDEX.md`。

## “直接输出”的精确定义

目标数据流是：

```text
warped image
  -> deterministic coarse absolute backward map + confidence
  -> confidence-protected coordinate flow matching（Stage 3）
  -> WARR 1/4-resolution recurrent refinement
  -> convex map upsampling
  -> final absolute backward map
  -> one grid_sample from the native warped image
  -> rectified RGB
```

因此，diffusion/Qwen 分支最终服务于二维 `velocity/backward_map` 预测；最终 RGB 不经过
Qwen VAE Decoder，也不从 Qwen 生成图取像素。Qwen 在 Stage 4 才作为可选的冻结全局条件特征
接入，且必须通过 hard-subset 增益门槛。

最终拓扑始终是 `coarse -> coordinate FM -> high-res refiner`。Stage 2 只是先把 refiner
接在 coarse 后预训练；Stage 3 插入 coordinate FM 时，`X0` 仍是 Stage-1 的真实预测 coarse
map，refiner 先复用/冻结，再做小学习率联合微调。

## 当前实现范围

当前已完成 Stage 0-5 所需的统一代码路径；这里仅表示模块、训练接口和契约已经落地，
不表示真实数据的任何质量 Gate 已通过：

- 唯一公共表示为 `[B,2,H_target,W_target]` 的绝对 backward map；通道顺序固定为 x、y；
- 全链 `align_corners=False`，配置不可覆盖；
- resize/crop/pad/flip 的 map 变换工具和回归测试；
- 严格 manifest，要求 `document_id`、`warp_severity` 和 `label_provenance`；
- CNN/FPN coarse map + per-pixel log-variance/confidence；新模型零初始化为 canonical map；
- confidence-protected residual coordinate flow matching：残差 proposal、置信度自适应噪声、
  8-block Coordinate Flow Transformer 与 4-6 步 Euler/ODE 更新；
- 单图 1/4 分辨率 feature warp + shared-weight WARR ConvGRU；
- RAFT-style convex map upsampling，取代最终 bilinear flow 上采样；
- H/V 图像导数结构分支、Jacobian cue、受限逐步 map residual；
- 冻结 Qwen-Image-Edit hidden/QK token 抽取、target/source split、DPT/FPN adapter 与
  confidence-aware gated fusion；Qwen pipeline 固定请求 `output_type=latent`，latent 随即丢弃；
- map、residual endpoint、低置信加权 velocity、sequence、uncertainty、H/V、straightness、
  mixed bending、anti-fold、Jacobian scale、valid-mask local SSIM 和 preserve losses；
- exact-pixel EPE/P95、fold、final-win、高置信区域破坏率指标；
- 分阶段 `coarse / warr / coord_fm / qwen / full_page` 训练、Gate receipt fail-closed、完整 checkpoint，
  以及 native-resolution 单次采样推理入口。
- Stage 0 数据审计会冻结 manifest、全部图像/map/mask payload SHA、document split、三随机种子与
  baseline identity；正式训练启动时重新哈希并拒绝任何静默数据漂移；
- Stage 5 采用 `low_page / structure_patch / full_page` 三视图混合。patch 只裁目标网格，source
  始终保留完整扭曲图，map 数值继续是完整 source 像素坐标，不做局部坐标重置；
- evaluator 固定输出逐样本表、置信度可靠性、运行时分解、误差热图、fold mask、线结构叠图与
  五联图；另提供不可覆盖的 Gate 1-5 receipt、多种子稳定性、重复推理和消融矩阵工具。
- Gate evaluator v3 固定实际 input/output work size；Gate baseline 必须同时匹配 manifest SHA 与
  全部验证 payload SHA、工作分辨率。冻结监督 prior 通过独立 backward-displacement→absolute-map 适配器评估，不会进入
  DocGrid-Flow checkpoint；Gate 5 的 Stage-2 确定性 baseline 会在同一全页集上重跑 1024×768。
- OCR scorer 会校验 OCR 与几何评估的 sample_id 集合，并把 CER/WER report 绑定到同一个
  checkpoint/manifest；Gate 5 不接受脱离该 report 的手填 OCR 数字。

真实 Qwen 权重没有写入 CP-DocFlow checkpoint，VAE Decoder 不在 forward 中调用。正式 Stage 2-5
仍必须按 `PROGRESS_PLAN.md` 的 Gate 顺序解锁；已有模块不能用 smoke test 绕过质量门禁。

## 数据契约

JSONL 每条记录至少包含：

```json
{
  "sample_id": "page-001-warp-a",
  "document_id": "page-001",
  "warp_severity": "medium",
  "label_provenance": "renderer_gt",
  "label_source": "dewarp_renderer_dataset_v2",
  "map_direction": "output_to_warped_source",
  "coordinate_convention": "absolute_source_pixel_xy",
  "warped_image": "images/page-001-warp-a.png",
  "rectified_image": "images/page-001-rectified.png",
  "backward_map": "maps/page-001-warp-a.npy",
  "valid_mask": "masks/page-001-warp-a.npy",
  "horizontal_structure": "structure/page-001-h.npy",
  "vertical_structure": "structure/page-001-v.npy",
  "boundary_structure": "structure/page-001-boundary.npy",
  "input_size": [1024, 768],
  "output_size": [1024, 768]
}
```

`backward_map[v,u]=(x_source,y_source)`，保存为输入图像像素坐标下的 float32 绝对坐标。
`.npy` 可为 `[H,W,2]` 或 `[2,H,W]`。训练配置必须显式列出
`allowed_label_provenance`；`raft_pseudo` 还必须登记 `label_checkpoint_sha256`。未确认的
`raft_pseudo`/`unknown` 会 fail closed，不能作为真实
`M*` 解锁 Gate 1 或 Gate 3。

现有旧 `metadata_*` flow 已追溯为 torchvision RAFT-Large 20-update 伪标签，而不是 renderer
GT。`cp_docflow.migrate_legacy_raft` 只允许将其转换为明确的 `raft_pseudo` exploratory manifest，
不能升级为 Gate GT。完整 SHA、解析 GT 重建流程和 Slurm 命令见
`docs/DATA_PROVENANCE.md`。

三个 structure 路径为可选项，但必须成组出现且同一 manifest 要么全部样本都有、要么全部没有。
没有显式结构标签时会从扭曲图像梯度生成确定性伪标签。

## 构建可审计的解析 GT

从平整文档 CSV 生成严格 document-disjoint 的解析 warp/map/structure 标签：

```bash
export DOCGRID_PYTHON=/path/to/training-env/bin/python
export DOCGRID_ANALYTIC_INPUT_CSV=/path/to/flat_documents.csv
export DOCGRID_ANALYTIC_OUTPUT=/path/to/new/docgrid_v2_analytic
bash slurm/docgrid_v2/00_render_analytic_gt.sh --partition=a100

export DOCGRID_TRAIN_MANIFEST="$DOCGRID_ANALYTIC_OUTPUT/manifests/train.jsonl"
export DOCGRID_VAL_MANIFEST="$DOCGRID_ANALYTIC_OUTPUT/manifests/val.jsonl"
export DOCGRID_TEST_MANIFEST="$DOCGRID_ANALYTIC_OUTPUT/manifests/test.jsonl"
bash slurm/docgrid_v2/00_audit_data.sh --partition=cpu
```

全量语料建议使用 Slurm array，所有 shard 成功后再合并；合并器会校验 shard 完整性、renderer
identity、manifest SHA、文档划分、样本唯一性、生成资产和每文档 variant 数：

```bash
export DOCGRID_ANALYTIC_SHARDS=16
export DOCGRID_ANALYTIC_SHARD_ROOT=/path/to/new/docgrid_v2_analytic_shards
bash slurm/docgrid_v2/00_render_analytic_gt_array.sh --partition=a100

export DOCGRID_ANALYTIC_MERGED_OUTPUT=/path/to/new/docgrid_v2_analytic_merged
bash slurm/docgrid_v2/00_merge_analytic_gt.sh --partition=cpu
```

Stage 0 应使用 merged output 下的三个 manifest，不能把单个 shard 当作完整数据集。
随后把同一套经过 Stage 0 的 train/val manifest 和 frozen contract 导出为
`DOCGRID_TRAIN_MANIFEST`、`DOCGRID_VAL_MANIFEST`、`DOCGRID_FROZEN_CONTRACT`；
Stage 1–4 配置会显式读取它们，解析后的绝对/相对路径写入运行记录。Stage 5 使用独立的
`DOCGRID_STAGE5_TRAIN_MANIFEST`、`DOCGRID_STAGE5_VAL_MANIFEST`、
`DOCGRID_STAGE5_FROZEN_CONTRACT`，必须对应它自己的全页 Stage 0 审计。

已知 `metadata_with_flow.csv` 的 `image` 虽然可作为 512×512 的解析 GT 源，但抽查及全量
元数据契约均表明它不是 Stage-5 原生全页源。不得将 512×512 图像拉伸为 1024×768 后称为
全页训练。容器入口 `00_render_full_page_array_a800.sh` 要求单独设置
`DOCGRID_FULL_PAGE_INPUT_CSV`，并在每个 shard 渲染前校验原图分辨率和 4:3 目标宽高比；随后依次
运行 `00_merge_full_page_cpu.sh`、`00_audit_full_page_cpu.sh`。Gate 5 默认只读取这套独立全页
validation manifest。

已核验的 116,016 条平整文档源及 array/merge/Stage0 环境变量模板位于
`examples/docgrid_full_analytic.env.example`；复制到仓库外的可写位置后修改输出根目录再 `source`。

解析几何只覆盖受控合成形变；最终 Gate 还需单独报告真实相机/域外集，不能用解析探针替代。

## CPU smoke

```bash
cd cp_docflow_v1
PYTHONPATH=src python -m cp_docflow.make_smoke_data \
  --output-dir tmp/smoke --train-count 4 --val-count 2
PYTHONPATH=src python -m cp_docflow.train_coarse --config configs/smoke.yaml
PYTHONPATH=src python -m cp_docflow.evaluate \
  --checkpoint tmp/smoke_run_contract_v1/best.pt \
  --manifest tmp/smoke/val.jsonl \
  --output-dir tmp/smoke_eval \
  --device cpu \
  --allowed-label-provenance synthetic_analytic
PYTHONPATH=src python -m cp_docflow.infer \
  --checkpoint tmp/smoke_run_contract_v1/best.pt \
  --image tmp/smoke/images/val-0004-warped.png \
  --output-dir tmp/smoke_infer --device cpu
```

完整 `coarse -> coordinate FM -> recurrent refiner` 图（用 `lite` 冻结特征源代替 20B Qwen，
但 adapter/fusion/velocity/refiner 拓扑完全相同）：

```bash
PYTHONPATH=src python -m cp_docflow.train_full --config configs/full_smoke.yaml
PYTHONPATH=src python -m cp_docflow.infer_full \
  --checkpoint tmp/full_smoke_run/best.pt \
  --image tmp/smoke/images/val-0004-warped.png \
  --output-dir tmp/full_smoke_infer --device cpu
PYTHONPATH=src python -m cp_docflow.evaluate_full \
  --checkpoint tmp/full_smoke_run/best.pt \
  --manifest tmp/smoke/val.jsonl \
  --output-dir tmp/full_smoke_eval --device cpu \
  --allowed-label-provenance synthetic_analytic
```

运行轻量回归：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

正式配置统一放在 `configs/docgrid_v2/`；Slurm 入口统一放在 `slurm/docgrid_v2/`。
Newest-plan 的逐项实现/证据状态见 `docs/REQUIREMENT_MATRIX.md`。
对“平台外层分配资源、内部使用 `srun --container-image`”的集群，直接提交
`slurm/docgrid_v2/container_jobs/` 中的 Bash 脚本，具体顺序见该目录 README；不要在容器里运行
会再次调用 `sbatch` 的登录节点 wrapper。
先执行 `bash slurm/docgrid_v2/00_audit_data.sh` 冻结数据，再执行
`bash slurm/docgrid_v2/01_train_coarse.sh`，然后在每个 reviewed Gate 通过后依次提交 `02` 到
`05`。`submit_multiseed.sh` 用 Slurm array 启动固定三 seed；`06_evaluate_gate.sh` 只接受冻结的
verified-GT evaluator 契约。没有 frozen contract、父 checkpoint、receipt 或本地 Qwen 权重时
脚本会拒绝提交。正式 Qwen 环境需安装 `.[qwen]`；当前 RAFT smoke 环境没有 `diffusers`，不能把
lite backend 结果当作 Qwen 结果。完整操作见 `slurm/docgrid_v2/README.md`。

Stage 4/5 还要求先运行 `bash slurm/docgrid_v2/00_validate_qwen.sh`。该作业用真实权重验证
hidden/QK target/source token 形状、有限值和 decoder-free 路径，并冻结本地 Qwen 配置 SHA；
缺失或过期 receipt 时 Stage 4/5 preflight 会拒绝提交。

Gate 5 几何评估会自动导出 `ocr_images.jsonl` 以及逐样本的 model/oracle/target
PNG。固定 OCR 引擎必须直接读取该清单中的 `model_image` 与 `oracle_image`；转录
CSV/JSONL 每行必须包含
`sample_id/reference_text/model_text/oracle_text/model_image_sha256/oracle_image_sha256`。
其中两个图像 SHA 必须原样复制自 `ocr_images.jsonl`，评分器会重新哈希图片并与几何
评估报告交叉校验。评分入口：

```bash
export DOCGRID_OCR_TRANSCRIPTS=/path/to/transcripts.jsonl
export DOCGRID_OCR_OUTPUT=/path/to/new/ocr_evidence
export DOCGRID_OCR_ENGINE=PaddleOCR
export DOCGRID_OCR_ENGINE_VERSION=3.0.0
export DOCGRID_GEOMETRY_EVALUATION=/path/to/gate_eval/metrics.json
export DOCGRID_GEOMETRY_PER_SAMPLE=/path/to/gate_eval/per_sample.csv
export DOCGRID_OCR_IMAGE_MANIFEST=/path/to/gate_eval/ocr_images.jsonl
bash slurm/docgrid_v2/07_score_ocr.sh --partition=cpu
```

`DOCGRID_OCR_ENGINE` 和 `DOCGRID_OCR_ENGINE_VERSION` 必须是实际固定值；`unknown`
会被拒绝。
