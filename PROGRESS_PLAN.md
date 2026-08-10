# DiffSynth GenWarp 当前进度与 CP-DocFlow 执行计划

最后更新：2026-07-31（主线由 v3.3 teacher/C4 residual 切换为 CP-DocFlow）

## 1. 当前决定

项目主线现切换为 **CP-DocFlow（Confidence-Protected Continuous Coordinate Flow for
Document Rectification）**。核心问题不再是“让 Qwen 先生成一张矫正 RGB，再从两张图之间估计
光流”，而是：给定单张扭曲文档，直接预测从 canonical output grid 指向原始扭曲图的
`backward_map`，最后只从原始高清输入采样一次。

目标推理链路固定为：

```text
warped image
→ deterministic coarse backward map Mc + confidence C
→ confidence-protected continuous coordinate flow matching
→ WAFT/RAFT-style high-resolution recurrent refinement
→ final backward map M
→ one grid_sample from the original high-resolution warped image
→ final rectified RGB
```

本文中“Qwen/diffusion 直接输出光流并得到最终矫正图”有严格含义：

- “光流”指文档矫正使用的二维 `backward_map`，或积分得到该 map 的二维 velocity；
- diffusion/coordinate-flow 分支直接预测的是 map/velocity，不是 RGB latent；
- Qwen-Image-Edit 仅作为候选的全局几何条件特征源，是否保留由阶段 4 的量化门槛决定；
- Qwen 的 VAE Decoder 不生成最终矫正 RGB，也不得成为最终文字像素来源；
- 最终 RGB 必须由 final map 对原始高清扭曲图执行一次 `grid_sample` 得到。

因此，“模型直接输出”描述的是最终几何预测责任，不是恢复旧的 Qwen RGB 生成链路。

## 2. 当前证据与问题定位

### 2.1 已确认的事实

- GT backward map 的 oracle warp 已人工验证通过，`backward_map + grid_sample` 的最终像素链路成立。
- 当前问题已经收敛为：如何仅从一张扭曲图像高精度预测 backward map。
- 当前 prior EPE 为 `5.7501`；v3.1 的最终输出没有超过 prior：

| 配置 | EPE | EPE P95 | 行 EPE | 直线度误差 | Fold rate | Final win rate |
|---|---:|---:|---:|---:|---:|---:|
| Prior | 5.7501 | 待阶段 0 补测 | 待阶段 0 补测 | 待阶段 0 补测 | 待阶段 0 补测 | 基准 |
| v3.1 Qwen on | 6.6009 | 12.8329 | 6.5550 | 0.1156 | 0.00446 | 0.3879 |
| v3.1 Qwen off | 7.2050 | 13.9506 | 7.1786 | 0.1265 | 待阶段 0 补测 | 待阶段 0 补测 |

这些数值继承旧数据与旧 manifest；其 `label_provenance` 尚未完成 Stage 0 审计，因此目前只是
historical reference，不是 verified-GT gate 结果。Stage 0 必须在来源确认并隔离的数据集上重算，
不得用这张表直接解锁 Gate 1 或 Gate 3。

这说明 Qwen 特征可能有相对收益，但 v3.1 整体仍会破坏 prior 已正确的区域。低分辨率 flow、
整幅坐标场无差别更新，以及 crop 训练与全页推理尺度不一致，是当前需优先验证的失败来源。
在确定性基线成立前，不把“Qwen 是否有效”当作主问题。

### 2.2 冻结的 v3.1 资产

- v3.1 已完成 20 epochs。workspace 与 data mirror 的同名 `epoch_0020.pt` 字节不一致；权威冻结
  checkpoint 明确选择 workspace 副本，不按文件名自动择取，也不允许用 mirror 副本替换：

| 角色 | 绝对路径 | Size (bytes) | SHA-256 | Best `line_epe` |
|---|---|---:|---|---:|
| **权威冻结基线** | `/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/diffusion2raft_unified_v3_1/runs/d2r_v3_1/unified/epoch_0020.pt` | 110,961,174 | `7f5f743de8994907b916182fd3d1c1e81c0015b5b53fb8bf5038ff4c2ad17fe5` | 4.15146515322 |
| 冲突副本，禁止作基线 | `/juicefs-algorithm/data/IPT/zhuochu_yang/diffsynth_genwarp/diffusion2raft_unified_v3_1/runs/d2r_v3_1/unified/epoch_0020.pt` | 110,957,462 | `826b8817dc6616e1d253ff4cfb24164f3575bf0607dcada1e2f95e9945ff8701` | 4.15439771302 |

- 加载冻结基线时必须同时校验绝对路径、size 与 SHA-256。data mirror 文件仅保留为 conflict
  evidence，不得用于复现报告、初始化新实验或 Gate 评估；`runs/` 不参与双目录同步，因此同步脚本
  不会也不应调和这两个 checkpoint。
- 现有 v3.1 代码、配置、checkpoint 和完整评估 JSON 作为只读 baseline/evidence 保留，不覆盖、
  不原地改造成 CP-DocFlow。
- CP-DocFlow 使用新的实验名、配置名、checkpoint 目录与 metrics schema，避免与 v3.1/v3.3 证据混写。

### 2.3 2026-07-30 新范式工程落地（DocGrid-Flow v2）

已新建独立目录 `cp_docflow_v1/`，不修改冻结的 v3.1/v3.3 checkpoint schema。当前已经完成
Stage 0-5 的**统一工程实现**；另外完成了真实平整文档输入上的小规模解析-GT/Stage-0/Stage-1
闭环，但不表示正式全量数据的任何质量 Gate 已通过：

- 固定 absolute backward map、x/y 通道顺序和 `align_corners=False`；
- 实现 map-aware resize/crop/pad/flip 与 native-resolution 单次采样；
- manifest 必填 `document_id`、`warp_severity`、`label_provenance`，train/val 文档泄漏会 fail closed；
- 实现不含 Qwen/diffusion 的 CNN/FPN coarse map + log-variance/confidence 基线、loss、评估和推理入口；
- 实现 confidence-protected `X0/Xt/velocity`、多步 Euler 坐标传输、1/4 feature-warp ConvGRU
  refiner、H/V/Jacobian cue、Qwen target/source token DPT/FPN adapter 与 gated fusion；
- 实现规范化的 `coarse/warr/coord_fm/qwen/full_page` 分阶段冻结策略、Gate receipt fail-closed、完整 checkpoint 与
  native-resolution final inference；Qwen VAE Decoder 不调用，Qwen 预训练权重不写入 checkpoint；
- 实现 Stage 5 `low_page/structure_patch/full_page` 三视图训练：patch 只裁 target window，source 仍为
  完整扭曲页，map 不重置为 patch 局部坐标；
- 实现 Stage 0 immutable 数据审计和 frozen contract：哈希 manifest 及全部 payload、检查 document
  split、记录 oracle/provenance/三 seed/baseline identity；正式训练前逐文件复核；
- 已追溯旧 `metadata_*` flow：它由 torchvision RAFT-Large 对 corrected/warped 图像对推理 20 次得到，
  是 `raft_pseudo` 而非 renderer GT；迁移工具固定记录生成器/权重/源 flow SHA 且禁止升级 provenance；
- 实现独立解析 GT renderer：从平整文档生成 perspective + smooth bend 的精确 absolute backward map、
  valid/H/V/boundary 标签和 document-hash split，并保存与渲染分辨率严格一致的 rectified target；
- 实现全量解析 GT 的 Slurm array 分片渲染和 fail-closed 合并：全局选择后按文档分片，合并时校验
  shard/report identity、renderer/source SHA、manifest SHA、生成资产、variant 完整性、样本唯一性和
  document split；Stage 0 只使用合并后的完整 manifest；
- Stage 1–5 的 YAML 与 Slurm preflight 现通过显式 `DOCGRID_*` 环境变量绑定同一份 Stage-0
  manifest/frozen contract、reviewed receipt 与 parent checkpoint；解析后的路径会冻结进运行记录，
  不再需要为了迁移数据根目录手改训练代码或配置；
- 严格隔离 Qwen adapter：Stage 1–3 不实例化、不保存未训练 adapter，Stage 4 才从无 adapter 的
  Stage-3 checkpoint 显式初始化并训练；兼容加载器只允许 adapter 前缀的历史额外/缺失 state，其他
  checkpoint 结构不匹配仍 fail closed；
- Slurm Stage 4/5 wrapper 现按 `DOCGRID_QWEN_HOST_MEM` 默认申请 192G host memory，以匹配冻结
  20B Qwen 的 CPU offload；确定性阶段仍默认 64G，用户可在提交时以显式 `--mem` 覆盖；
- Stage 2 已按前 20% WARR-only、后 80% 小学习率联合微调实现；Stage 3 已按前 70% velocity-only、
  后 30% 小学习率完整链联合微调实现，optimizer 在启动时覆盖全部分段参数；
- 实现 Gate 1-5 全部硬规则、固定可视化产物、置信度可靠性、几何质量、L1/PSNR/SSIM、字符边缘、
  表格连通率、运行时指标、多种子聚合、精确 repeatability 和命名消融矩阵；Gate 5 几何评估会导出
  model/oracle/target 全量 PNG 与 SHA-bound `ocr_images.jsonl`，OCR CER/WER report 必须绑定同一
  checkpoint/manifest/payload/sample set 和逐图 SHA，拒绝改图、未知引擎版本、手填或跨实验复用数字；
- 95 个 CPU 回归测试通过，覆盖坐标、target-window、bilinear/convex、mask-aware resize、数据/provenance、confidence calibration、
  empty-valid fail-closed、gate/exploratory 隔离、模型、指标、checkpoint 与“最终 RGB 只从 native 原图采样一次”契约；
- analytic smoke 闭环已跑通：4 train + 2 val、1 epoch CPU，`EPE=0.709829`、`P95=1.407835`、
  `fold_rate=0`、work-size oracle RGB L1 `0.000320`；这些数字只验证工程链路，
  报告固定为 exploratory，`synthetic_analytic` 不得声明 gate eligibility。
- 完整轻量图另完成 1 epoch smoke：`coarse_epe=0.604273`、`final_epe=0.787591`、
  `P95=1.473182`、`fold_rate=0`、高置信区域破坏率 `0.015489`。单 epoch final 暂低于 coarse，
  这只证明端到端训练/保存/恢复可运行，明确不构成 Stage 2/3/4 精度证据。
- Stage 5 mixed-view smoke 已记录三种 view、expanded config、run manifest、逐 loss 指标与 checkpoint
  `training_seed`；同一 checkpoint 的 15 组 runtime ablation 及两次 bit-exact 推理均已跑通。这些仍是
  synthetic engineering evidence，不是 Gate 结果。
- 40 个真实平整文档经解析 renderer 生成 20/13/7 的 train/val/test，Stage 0 确认
  `verified_gt_only=true`、document split 无泄漏、三组 fold rate 均为 0；其 64×64 Stage-1 训练探针
  完成 10 steps 并评估为 `EPE=0.596250`、`P95=1.165082`、`fold_rate=0`。这证明真实文件路径到
  checkpoint/evaluator 的工程闭环，不是全分辨率质量 Gate。
- 已重新核验全量候选 CSV `upwarp_img_1in10_white_0511/metadata_with_flow.csv`：共 116,016 行，
  `image` 是平整 GT 源，`edit_image` 是旧扭曲图，`flow_gt_path` 仍是 RAFT pseudo；新增可复制的
  `examples/docgrid_full_analytic.env.example` 用于 array render→merge→Stage 0 的正式环境绑定。
- 本地进一步执行了真实 CSV 的两 shard CPU 路径：40 个 128×128 文档合并后仍为 20/13/7，
  Stage 0 为 `verified_gt_only=true`、document-disjoint、全 split 零 fold；证据位于
  `cp_docflow_v1/tmp/analytic_sharded_cpu_probe_0731/`，frozen contract SHA-256 为
  `b48de12a7002392f48394fda4f2f6fdb12e2ffc1bf16989df6c83073633cc7cc`。
- 针对实际“外层平台分配 + 内层 `srun --container-image`”方式新增
  `slurm/docgrid_v2/container_jobs/`：不再容器内嵌套 `sbatch`，覆盖 64-shard A800 渲染、CPU
  merge/audit、Qwen runtime validation、Stage 1–5、Gate 评估/人工签收与 OCR。
- 已补齐正式 Gate baseline 链路：冻结的 259999 强监督 TorchScript prior 由独立适配器从旧
  512-square backward displacement 转换为 `align_corners=False` absolute map，再在同一冻结 val
  manifest 上生成 evaluator-v3 指标；Stage-0 audit 会交叉验证 checkpoint/config/metrics SHA、manifest
  SHA 和工作分辨率。Gate 2/3/4 自动选择前一阶段同集评估，Gate 5 则要求 Stage-2 WARR 在同一全页
  manifest 上以 1024×768 重跑，receipt 拒绝不同 canvas 的伪比较。
- `cp_docflow_v1/docs/REQUIREMENT_MATRIX.md` 已逐项区分 newest plan 的“工程已实现”与“正式证据待跑”；
  当前模型、训练、评估和 fail-closed Gate 契约已落地，但全量 Stage 1-5 三 seed、真实 Qwen receipt、
  A1-A20 正式消融、OCR 与最终人工质量验收仍须由 Slurm 结果完成。

当前仍未完成的是**全量正式数据与真实相机域的 Stage 0 冻结**：旧 116,016 行 flow 已确认是
RAFT pseudo label，不能直接用作 Gate GT；仍需用解析/renderer 来源为全量训练集重建 exact map，建立
真实 renderer-GT 或真实相机域外测试集，并补齐冻结 prior 指标。40 页探针只授权工程验证，不授权
正式三 seed Stage 1-5 质量结论。

工程代码入口已经就绪，但当前 `RAFT_flow` 环境未安装 `diffusers`；真实 Qwen-Image-Edit hidden/QK
feature capture 需在 Slurm 的 Qwen 环境用本地 20B 权重完成形状、显存和运行时验证。lite backend
只验证 adapter/fusion/topology，不得写成真实 Qwen 实验结果。

## 3. v3.3 teacher/C4 路线：冻结为 historical baseline

原 v3.3 路线围绕固定 RAFT teacher、C4 朝向选择与有界 residual capacity 展开。其研究 teacher 为：

```text
/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/399999_raft_unwarp.pt
SHA-256: e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5
```

历史实验的关键结论继续保留：

| Slurm job | 历史实验 | 核心结果 | 归档结论 |
|---:|---|---|---|
| 78888 | 259999 teacher capacity | 大旋转下出现约 90° 朝向混淆 | 该 production teacher 对旧路线不合格 |
| 78920 | 399999 teacher capacity | 原始朝向更好，但大旋转混淆仍存在 | 不再扫描中间 checkpoint |
| 78932 | rotation-only oracle C4 v1 | rotation EPE `218.5000 → 2.1680` | 主要失败是离散 C4 朝向混淆 |
| 78933 | canonical-frame oracle C4 v2 | EPE `2.16799`；solver/trainable `0.983346/0.980576` | canonical frame 正确，6 次 solver 不足 |
| 78943 | canonical solver sweep v3 | 12 次 solver/trainable `0.995784/0.992237` | 12 次是旧 rotation-only aggregate 的最小合格配置 |
| 78946 | canonical full-geometry v4 | EPE `3.57260`；solver `0.993225`；overflow `0.008614` | aggregate solver 通过，其他门槛未全过 |
| 78953 | full-geometry capacity grid v5 | `12/24 × 24/32/40 px` 六格完成 | 发现 ±45° 路由边界 catastrophe |
| 78979 | full-geometry best-of-C4 v6 | nearest/best-EPE `299/300`；六格 best-EPE 与 capacity-aware 标签完全一致 | 严格逐桶仅 `24×40` 零失败 |

其中 job 78936 只是提交命令漏写 `.sh` 导致 exit 127，没有产生实验结果。v6 报告 SHA-256 为
`abf55cc22ea65665d175563c73d18ce993d7401923c4afe4e824073900655176`。

这些结论只描述旧 teacher/residual 路线的 diagnostic upper bound。best-of-C4 标签来自部署时不可获得
的 injected rotation 或 GT flow，`24 solver × 40 px cap` 也只是旧 residual 表示的容量上界。
它们不得：

- 生成或替代 CP-DocFlow 的 production approval；
- 解锁 CP-DocFlow 的 smoke/training；
- 成为 CP-DocFlow 阶段 0-6 的进入条件；
- 被误解为真实 image-only orientation router 已完成；
- 被直接复用为 CP-DocFlow 的 solver 次数、残差范围或网络迭代数。

原计划中的 v7 structural-safety diagnostic 和后续真实 C4 router 研发不再是“当前唯一下一步”，
也不 gate 新主线。除非明确开展历史复现或独立对照，不再运行以下旧流水线：

```text
slurm/v33_pipeline/diagnostic_teacher_399999_best_of_c4_structural_safety_v7_1gpu.sh
slurm/v33_pipeline/01_teacher_capacity_1gpu.sh
slurm/v33_pipeline/02_teacher_smoke_8gpu.sh
slurm/v33_pipeline/03_teacher_train_8gpu.sh
```

正式 config 与 `slurm/v33_pipeline/common.sh` 中既有 teacher 绑定保持历史状态；不要为 CP-DocFlow
静默切换或覆盖它们。若未来把 C4 机制作为独立模块重新引入，必须重新建立 image-only 证据和
CP-DocFlow 自身的 contract，而不能沿用 GT diagnostic approval。

## 4. CP-DocFlow 模型边界

### 4.1 数据与坐标定义

对矩形输出位置 `(u, v)`，final backward map 定义为：

```text
M[v, u] = (x_w, y_w)
I_rectified = grid_sample(I_warped, M)
```

全项目固定以下约定：

- 主标注保存为输入图像像素坐标下的 `float32` 绝对坐标；
- `grid[..., 0]` 始终为 `x`，`grid[..., 1]` 始终为 `y`；
- EPE 在像素坐标下计算；进入 `grid_sample` 前才转换到 `[-1, 1]`；
- 统一使用 `align_corners=False`；
- 位移 `F=M-P` 只作为网络内部变量，`P` 为 canonical output grid；
- resize、crop、padding、flip 必须对图像和 map 应用同一个坐标变换；
- train、validation、full-page inference 必须使用相同的 map 语义。

当前旧数据中的 map 已确认来自 corrected↔warped 图像对之间的 torchvision RAFT pseudo label，而不是真正的
解析/renderer GT。迁移时必须为每个样本增加必填字段
`label_provenance`，至少区分 `analytic_gt`、`renderer_gt`、`raft_pseudo` 和 `unknown`。审计证据还需
能够追溯 map 的生成方向、坐标约定和 source/checkpoint identity；无法确认的记录一律归为
`unknown`，不得猜测为 GT。

Gate 1/3 的 `M*` 与 gate metrics 只能来自已验证的 `analytic_gt` 或 `renderer_gt`。`raft_pseudo` 可在
显式标记、独立配置和独立报告下用于探索性训练或对照，但不能与 GT 混合统计，也不能作为 `M*`
解锁 Gate 1 或 Gate 3；`unknown` 不得进入任何 gate 数据集。

### 4.2 模块职责

1. **Deterministic coarse map**：CNN/ConvNeXt/Swin + FPN 从单图预测 `Mc` 与 confidence/log-variance。
2. **Coordinate flow matching**：从受 confidence 保护的 `Mc` 出发，仅给低置信区域足够的生成搜索
   空间；学习 map-space velocity，并用 4-6 步 ODE/Euler 更新，而不是从纯噪声重建整幅 flow。
3. **High-resolution recurrent refinement**：将当前 map 对应的原图特征 warp 到 canonical 域，
   用共享权重 ConvGRU/WAFT 式更新预测受限小残差；单图任务不照搬双帧 RAFT all-pairs correlation。
4. **Optional Qwen conditioning**：冻结 Qwen 图像特征，经 DPT/FPN 与 CNN 局部特征门控融合；
   final head 仍输出二维 map/velocity。只有冻结特征通过阶段 4 gate 后，才允许评估少量 LoRA。
5. **Rendering**：只在所有几何预测完成后，对原始高清输入执行一次 `grid_sample`；中间 preview
   可用于特征更新或辅助损失，但不得成为最终 RGB。

### 4.3 训练约束

- 训练初期以 map、sequence、velocity loss 为主；RGB warp loss 仅作低权辅助。
- 结构损失至少覆盖 gradient、H/V、bend、Jacobian、confidence 和 preserve。
- 若 warped/rectified 图存在阴影或光照差异，不允许 RGB loss 通过错误位移追逐颜色。
- 每个 recurrent 中间 map 都计算 sequence loss；单步 residual 必须受限。
- Stage 3 的 `X0` 始终由 Stage 1 网络真实预测的 `Mc` 构造，不得用 GT coarse map、Stage 2 refined
  map 或 pseudo map 替代。
- 高置信区域使用更小噪声、soft anchor 和 preserve loss，但不永久硬锁定。
- 关键结果至少报告 3 个固定随机种子的均值与标准差。

## 5. 最终验收指标

以下指标以当前 prior EPE `5.7501` 为基准；阶段 0 会补齐 prior 的其余指标并冻结 evaluator：

| 指标 | 最低验收目标 | 理想目标 |
|---|---:|---:|
| Final EPE | `≤ 5.18`，相对 prior 至少下降 10% | `< 4.50` |
| EPE P95 | 相对最终确定性基线下降至少 10% | 下降至少 15% |
| Final win rate | `≥ 0.65` | `≥ 0.75` |
| 直线度误差 | `≤ 0.10` | `≤ 0.08` |
| Fold rate | `≤ 0.0045`，且不高于确定性基线 | `< 0.002` |
| 高置信区域破坏率 | `< 5%` | `< 2%` |
| OCR CER | 不高于 oracle CER 加 1 个百分点 | 尽量接近 oracle |
| 推理步数 | 4-6 次坐标传输 + 4-6 次局部更新 | 不降精度时继续减少 |

“高置信区域破坏率”定义为：coarse map 误差已经小于 1 px，但 final map 误差增加超过 1 px 的
像素比例。该指标与 EPE、P95、fold、直线度共同决定是否通过，不能只用平均 EPE 覆盖局部破坏。

## 6. 阶段 0-6 与 Go/No-Go 门槛

| 阶段 | 当前状态 | 建议时长 | 主要交付物 | 进入下一阶段的硬门槛 |
|---|---|---:|---|---|
| 0. 评估、provenance 与坐标固化 | 92 tests、旧标签取证、解析 renderer/分片合并、40 页 audit 与 prior-eval/绑定代码已完成；全量/真实域实际产物仍待 Slurm | 2-3 天 | 统一 evaluator、带 `label_provenance` 的 manifest、oracle/transform tests、完整 baseline 报告 | 所有坐标测试通过；GT/pseudo 已分离；Gate 1/3 的 verified-GT 集、prior/v3.1 指标与 3 seeds 冻结 |
| 1. 确定性粗 map | 工程基线和 audited 40-page CPU probe 已完成；正式全量训练仍被 Stage 0 gate 锁定 | 1-2 周 | `det_coarse` checkpoint + calibrated confidence | EPE `≤5.75`，fold 不高于 prior/冻结 baseline |
| 2. 确定性循环精修预训练 | 模块/训练入口已实现；待 Gate 1 后正式训练 | 1-2 周 | 在 coarse 后预训练的 `det_refine` checkpoint，4/6-step 和分辨率对照 | 相对 coarse EPE 至少下降 `0.3 px`、win rate `≥0.60`；EPE/P95/直线度同时改善且 fold 不升 |
| 3. 置信度坐标传输 | 模块/损失/ODE 已实现；待 Gate 2 | 1-2 周 | 插在 coarse/refiner 之间的 `coord_fm` checkpoint，固定 4-6 步 sampler | 完整 `coarse→FM→refiner` 相对 det_refine EPE 下降 `≥5%` 或 `≥0.2 px`；win rate `≥0.60`；高置信破坏率 `<5%`；3 seeds 稳定 |
| 4. Qwen 条件融合 | 实际后端/adapter 已实现；待 Gate 3 与 Qwen 运行环境 | 1 周 | `qwen_cond` checkpoint 与 CNN/Qwen fusion 消融 | 全局至少改善 2%，hard/重形变子集至少改善 5%，且全局指标不退化 |
| 5. H/V 与全页联合微调 | H/V 与三视图 mixed-view 工程已实现；正式训练待前序 Gate | 1-2 周 | 完整模型、全页高分辨率报告 | 达到第 5 节全部最低验收目标 |
| 6. 消融与论文结果 | 工具/15 组 smoke 已完成；正式三 seed 表待训练 | 1 周 | 多 seed 主表、消融表、效率/失败案例/可视化 | 结论可重复，模块贡献清晰，结果可审计 |

### 6.1 Gate 解释

- **Stage 0 不通过**：只要 label provenance 或 map 方向仍不明确，就不得用相关 map 计算 Gate 1/3，
  也不得把 RAFT pseudo label 写成 `M*`。先完成数据隔离与 manifest 修复。
- **Stage 1 不通过**：只在冻结的 verified-GT gate set 上判定；若未达到门槛，停止加入 diffusion、
  Qwen 或更深 GRU，先排查数据变换、输出分辨率、全页上下文、模型容量与数据划分。
- **Stage 2 不通过**：若只降训练 EPE 或平均 EPE，却恶化 P95、line/bend 或 fold，不算通过；优先
  排查 crop/full-page 尺度差，而不是增加 GRU 深度。
- **Stage 3 不通过**：只在同一个 verified-GT gate set 上判定；不进入 Qwen 融合，先检查 confidence
  calibration、residual 分布、采样方差与高置信区域破坏率。diffusion 必须证明相对确定性模型的
  独立增益，pseudo-label 指标不能代替该证据。
- **Stage 4 不通过**：最终模型删除 Qwen，保留更轻量的全局分支或纯 CNN；不得因已投入算力而放宽门槛。
- **Stage 5 不通过**：不以“视觉上更好”替代统一 evaluator；按失败指标回退到对应模块修复。

阶段编号表示研发顺序，不表示最终模块拓扑。最终拓扑始终固定为：

```text
Stage 1 predicted coarse Mc
→ Stage 3 coordinate flow matching
→ Stage 2 high-resolution refiner module
→ final map M
```

Stage 2 暂时采用 `predicted Mc → refiner` 预训练可复用的 high-resolution refiner。Stage 3 随后把
flow matching 插到两者之间：`X0` 始终从 Stage 1 的 predicted `Mc` 构造，Stage 2 refined output
绝不作为 `X0`。Stage 3 初期冻结并复用 Stage 2 refiner，先训练 coordinate velocity；稳定后再以
小学习率联合微调 FM 与 refiner，并以完整 `coarse→FM→refiner` 链路对比冻结的 `det_refine` baseline。

## 7. 立即执行任务

严格按以下顺序推进；40 页解析探针不替代全量 Stage 0 和正式 Gate：

1. 冻结并登记 v3.1/v3.3 代码、配置、checkpoint、完整 JSON、关键报告 SHA 和可视化，不覆盖旧资产。
2. 建立 CP-DocFlow 独立实验命名、config schema、output root 与 evaluator 版本标识。
3. 旧数据已确认为 corrected↔warped RAFT pseudo labels；迁移器已固定 `raft_pseudo`、map 方向和
   source/checkpoint identity。下一步只将它用于独立 exploratory 对照，不得混入 Gate GT。
4. 在 verified-GT 集上补齐 prior 与 v3.1 的 EPE P95、line、edge、fold、OCR CER、win rate、
   高置信破坏率和分子集指标；pseudo 结果单独报告。
5. GT oracle warp 与解析 renderer 已固化为自动测试，并覆盖 resize、crop、padding、flip、x/y 分量缩放、
   `align_corners=False` 与 full-page round trip。
6. 按 `document_id` 或原始来源审计 train/validation/test split，禁止同一文档的不同形变跨集合。
7. 固定三个随机种子与 verified-GT 验证集；每次实验统一输出 `config.yaml`、`metrics.json`、`per_sample.csv`、
   flow/map visualization，以及 `warped / prior / final / GT / oracle` 对比图。
8. deterministic coarse-map + uncertainty baseline 已实现；阶段 0 全部通过后启动正式三 seed 训练。
9. 共享权重 1/4 refiner 已实现；Stage 1 通过后在 predicted `Mc` 后预训练 4 次，再比较 6 次与
   `1/8、1/4、1/2` 分辨率。
10. confidence-protected coordinate FM 已实现；Stage 2 通过后在 `Mc` 与冻结 refiner 之间正式训练；
    `X0` 只来自 Stage 1 predicted `Mc`。先训 FM，再联合微调，并固定 4-6 步推理。
11. 冻结 Qwen-Image-Edit token 后端与 adapter 已实现；只有 Stage 3 receipt 后入口才允许正式启动，
    Stage 4 通过后才允许少量 LoRA 试验。
12. H/V 结构分支已实现；前序 Gate 通过后开展双尺度训练与真实全页联合微调，并完成全部消融。

## 8. 数据划分、报告与消融约束

测试集至少按以下维度分组报告：轻/中/重形变，小字/普通字号，中文/英文/数字/表格，页面边缘/
中心，阴影/模糊/低对比度，常规/超高分辨率，以及合成/真实域外数据。

必须完成的核心消融包括：

| 编号 | 对照 |
|---|---|
| A1 | coarse map vs coarse + recurrent refine |
| A2 | 全图加噪 vs 低置信区域加噪 |
| A3 | 无 guidance vs 固定 guidance vs 时间衰减 guidance |
| A4 | 直接回归 vs residual diffusion vs continuous coordinate flow matching |
| A5 | CNN only vs Qwen only vs concat vs gated fusion |
| A6 | 无 H/V vs H only vs H+V |
| A7 | `1/8` vs `1/4` vs `1/2` refine |
| A8 | 1、3、4、6、10 个 coordinate transport steps |
| A9 | crop training vs dual-scale training vs full-page fine-tuning |
| A10 | 无 preserve loss vs preserve loss |

任何关键结论均需保存 per-sample 结果、3-seed 均值/标准差、失败案例与实现 identity，避免只按单次
aggregate 指标选择模型。

## 9. 重要历史文件与新计划入口

- CP-DocFlow 目标与方法说明：`Diffusion2RAFT_Plan_and_Goals.md`
- CP-DocFlow 完整工程入口：`cp_docflow_v1/README.md`
- Qwen 输入、token adapter、velocity decoder 与 recurrent refiner 逐层设计：
  `cp_docflow_v1/docs/QWEN_IMAGE_EDIT_FLOW_ARCHITECTURE.md`
- v3.1 结构与训练说明：`diffusion2raft_unified_v3_1/README.md`
- 冻结的 v3.3 pipeline 说明：`slurm/v33_pipeline/README.md`
- 78943 solver sweep：
  `diffusion2raft_unified_v3_1/runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_solver_sweep_v3/candidate-399999.json`
- 78946 full geometry：
  `diffusion2raft_unified_v3_1/runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_v4/candidate-399999.json`
- 78953 capacity grid：
  `diffusion2raft_unified_v3_1/runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_v5/candidate-399999.json`
- 78979 best-of-C4：
  `diffusion2raft_unified_v3_1/runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_v6/candidate-399999.json`

这些历史报告可以作为 failure-mode 分析或对照，但不拥有 CP-DocFlow gate 的批准权限。

## 10. 双目录同步规则

权威编辑目录：

```text
/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp
```

data 镜像：

```text
/juicefs-algorithm/data/IPT/zhuochu_yang/diffsynth_genwarp
```

每次代码或计划更新完成并检查差异后运行：

```bash
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/scripts/sync_code_mirror.sh --v31
```

同步方向固定为权威编辑目录到 data 镜像。`runs/`、checkpoint、Slurm logs 和缓存不参与代码镜像；
历史 evidence 不因代码同步而重写。同步前必须确认改动范围，当前文档切换本身不授权修改任何训练代码。
