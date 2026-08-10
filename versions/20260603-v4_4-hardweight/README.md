# V4.4 — 难度加权 + 梯度累积 — 2026-06-03

## 一句话

继承 V4.2 全部架构（双图跨段 Q/K，**不改任何网络结构**），针对过拟合诊断定位的真实瓶颈——**难样本(大位移)欠拟合 + 训练未收敛**——做两处纯训练侧改进。

## 决策依据：high-only 长训诊断（log 74932）

| 实验 | high <5px | 含义 |
|------|----------|------|
| 18 样本混训 3000 步 | 53% | high 与 low/mid 抢容量/步数，欠拟合 |
| 6 个 high 单训 5000 步 | **99.5%** | **high 完全能学到位（含 168px 极端样本）** |

**结论：不是架构/容量/搜索范围瓶颈（V4.3 扩 radius 已证伪），而是难样本欠拟合 + 收敛慢。** high 收敛远慢于 low/mid，混训时被简单样本占满有效梯度步数。

## V4.4 两处改进（均不破坏单模型约束）

### 1. 难度加权 loss
按样本 zero-flow EPE（flow_gt 平均模长）给整体 loss 加权：

```
w = clamp((zero_epe / ref)^alpha, max)
total_loss = w * (lambda_flow·L_flow + L_grad + lambda_warp·L_warp)
```

参数（默认）：`alpha=1.0, ref=24.0(数据集中位数), max=3.0`

权重效果（验证过）：

| 层 | zero_epe | 权重 w |
|----|---------|-------|
| low | 7.7px | 0.32（简单样本降权）|
| mid | 24px | 1.00（中位数不漂移）|
| high | 31-59px | 1.3~2.5（难样本升权）|
| 极端 | 168px | 3.0（被 clamp，不爆炸）|

→ 难样本拿更多有效梯度，直击 high 欠拟合。`alpha=0` 时 w=1 完全等价 V4.2。

### 2. 梯度累积接线（修 bug）
V4.2 的 `--gradient_accumulation_steps` 是**死参数**：能传 CLI 但没传给 `launch_training_task`，所以从不生效（这也是之前 batch=1 震荡一直没改善的原因之一）。

V4.4 在 `main()` 的 `launch_training_task(...)` 调用里接线：
```python
gradient_accumulation_steps=args.gradient_accumulation_steps
```
`launch_training_task` 本身完整支持（`Accelerator(gradient_accumulation_steps=...)` + `accelerator.accumulate(model)`），只是 V4.2 没传。默认设 4 → 8卡×4 = 等效 batch 32，降噪 + 加速收敛。

## 文件结构

```
20260603-v4_4-hardweight/
├── README.md
├── train_flow_head_v4_4.py        # 训练（+难度加权 +梯度累积接线）
├── qwen_image_flow_v4_4.py        # 推理（与 V4.2 同逻辑，换模块文件名）
├── utils/flow_head_v4_4.py        # 模型（= V4.2 架构，未改）
├── overfit_samples.csv
└── scripts/
    ├── train_v4_4_layer36.sh      # alpha=1.0, grad_accum=4
    └── flow_v4_4_sample.sh
```

**架构 + ckpt 与 V4.2 完全兼容**（V4.4 没动网络）：V4.2 ckpt 可加载到 V4.4 续训。

## 怎么跑

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp
bash versions/20260603-v4_4-hardweight/scripts/train_v4_4_layer36.sh
```

ckpt 输出：`flow_head_v4_4_ckpts/20260603-v4_4-hardweight-layer36/`

## 日志怎么看（验证两处改进生效）

训练日志每 100 步打印（新增 `w=` 字段）：
```
[step N][K=warped] L_flow=.. L_grad=.. L_warp=.. w=1.85 total=.. | EPE=.. (zero=..) <1px=.. <3px=.. <5px=.. | pred_max=.. std=..
```

- `w=` 应随样本难度变化（简单 ~0.3，难 ~2-3）→ 确认难度加权生效
- 相比 V4.2，**震荡 std 应明显下降**（梯度累积降噪）
- **关键看 high 难度区间的 EPE/`<5px` 是否比 V4.2 提升更快**（难度加权的目标）

## 预期 & 验证

预期：全量训练时 high 层从 53% 拉到 80%+，整体逼近 90%。

**最快验证（推荐）**：用 V4.4 的难度加权跑 18 样本过拟合诊断（`overfit_diagnose.py` 加 `--hard_weight_alpha 1.0`），看 high 层能否在同样步数内追上 low/mid。若能，证明难度加权有效，再投入全量训练。

## 调参建议

- `alpha=1.0` 起步。若 high 仍慢，调 `alpha=1.5~2.0`（更激进偏向难样本），但注意别让 low/mid 退化。
- `grad_accum=4` 起步（等效 batch 32）。显存够可加大；想更快迭代可设 2。
- 两个改动可独立开关：`alpha=0` 关难度加权、`grad_accum=1` 关梯度累积，便于消融哪个贡献大。

---

## 修复记录（2026-06-03 晚）— 难度加权尺度 bug

**首版 V4.4 训练（log 75403）比 V4.2 更差**（40k 步 <5px 40.7% vs V4.2 60.7%），诊断发现是难度加权两个 bug：

1. **尺度 bug**：用 `flow_gt_r`（H/8 + 数值缩放 0.0625）的 EPE 算 w，但 `ref=24` 是原图尺度中位数 → w 全部 <0.3 → 难样本没被突出，反而**整体降权**，拖慢收敛。
2. **无归一化**：w 不做均值归一化 → 整体 loss 尺度漂移。

**修复**：
- 改用**原图尺度** zero_epe（缩放前的 `flow_gt_t`）算 raw_w
- 加 **EMA running 均值归一化**：`w = raw_w / running_mean`，让权重围绕 1 分布——只改"相对难度分配"（难样本升权、简单样本降权），不改整体 loss 尺度
- clamp 到 `[1/3, 3]`

修复后权重验证（分层样本）：low 平均 w=0.80（降权）、high 平均 w=2.18（升权）、极端 168px clamp 到 3.0 ✓

3. **grad_accum 4 → 2**：grad_accum=4 让有效优化步数变 1/4，收敛太慢；降到 2 平衡降噪与收敛速度（等效 batch 16）。

**对比 V4.2 时注意**：V4.4 用 grad_accum=2，"step 数"要除以 2 才是参数更新次数。按参数更新次数对齐才公平。

---

## 新增（2026-06-03）— VAE round-trip 域对齐

**动机**：训练时 main 图 = GT 矫正图（清晰），推理时 main 图 = Qwen 50步生成的 corrected_low（有 VAE 损耗+可能改字）。这个 train/infer 域差让模型在真实图推理时失准——是"真实图推理效果不如训练集指标"的主因之一。

**方法**：训练时按 `--vae_roundtrip_prob`（默认 0.5）把 main 图换成 `corrected_vae`（GT 经 VAE 编码-解码 round-trip 的版本，已全量预生成 116k 张）。corrected_vae 带 VAE 损耗，比纯 GT 更接近推理时的 corrected_low → 缩小域差。

**实现**：
- 数据：合并 `metadata_with_corrected_vae_part/*.csv` → `metadata_with_vae.csv`（含 `corrected_vae_path` 列，116016 行全部有效）。
- `FlowDataset.__getitem__`：按概率把 `data["image"]` 替换成 corrected_vae（resize 到 main 图尺寸，**只换 main，warped 和 flow_gt 不变**）。
- 50% 概率混入：保留一半纯 GT（信息更干净），一半 vae 版（贴近推理域）。

**验证**：corrected_vae(1024) resize 到 512 与 GT 对齐 ✓；GT vs vae MSE=1.2（确有 round-trip 损耗）✓。

**为什么这对"真实图推理改善"可能比难度加权更直接**：难度加权解决的是难样本欠拟合（影响训练集 high 区间指标）；域对齐直接缩小推理时 corrected_low 的分布差异，对真实图视觉效果的提升更对症。

## V4.4 最终配置（三件套）

```
--hard_weight_alpha 1.0       # 难度加权（修复尺度bug后，EMA归一化）
--gradient_accumulation_steps 2  # 梯度累积（修死参数 + 等效batch16降噪）
--vae_roundtrip_prob 0.5      # VAE round-trip 域对齐（缩小推理域差）
```
三者均不破坏单模型约束，可独立 ablation（设 0 关闭对应项）。

---

## 🔴🔴 致命根因修复 — CorrBlock 缺 identity base grid（2026-06-10）

**这是从 V4 延续到 V4.4 全部五个版本的结构性 bug，是"水波纹 / 推理崩 / 训练侧改进全无效"的真正根因。** 三件套（难度加权/梯度累积/VAE对齐）全是训练侧改进，治不了它——这解释了为什么 V4.4 训到 40 万步（≈20 万更新）仍与输入几何无关。

### 现象（触发排查）
V4.4 训到 step-404000 推理仍失败：几乎平整的扫描页被 warp 出 mean≈24-30px 大波浪，输出 flow 与输入几何几乎无关；中间扩散 corrected_low 反而比最终 flow warp 更平整（DiT 懂几何，flow head 读不出）。

### 根因
[utils/flow_head_v4_4.py](utils/flow_head_v4_4.py) 的 `CorrBlock.__call__` 计算相关查表坐标时：

```python
# 错误（原代码）：把"位移量 flow"直接归一化当成"绝对采样坐标"，丢了 identity 网格 coords0
centroid_norm = flow / [W-1, H-1] * 2 - 1
sample_coords = centroid_norm + offset_norm
```

标准 RAFT 的查表中心应是 **`coords0(像素自身坐标) + flow(位移)`**。漏掉 `coords0` 后：
- **flow=0 时每个像素都去 warped 特征图左上角 (-1,-1) 附近采样**（而非自己对应位置）
- correlation volume 与几何位置完全脱钩 → warped 几何信息唯一入口被堵死
- 模型退化成"仅凭 `context_encoder(corrected)` 纯 CNN 通路回归一个平均扭曲场"
- 训练时记住 GT 外观（过拟合，故 overfit 诊断曾 99.5%），推理换 corrected_low 即崩

### 修复（已数值验证）
```python
# 正确：coords = coords0 + flow，再加搜索窗 offset，统一归一化
self.coords0 = stack([gx, gy])           # __init__ 建特征分辨率 identity grid
coords = self.coords0 + flow.permute(...)  # 像素自身坐标 + 位移
sample_norm = (coords + delta) / [W-1,H-1] * 2 - 1
```

| 验证指标 | 修复前 | 修复后 |
|---------|-------|-------|
| 逐像素采样命中正确位置 | **0.0%** | **93.8%** |
| flow=0 center-tap 自匹配（应≈1.0）| 0.015（≈随机）| **1.0000** |
| 合成单样本 overfit EPE | 学不出 | **2.56px → 0.056px** |

### 顺手修复（minor）：dpt_ctx 接回
原 `forward` 里 `F_Q, F_K, _ = self.dpt_heads(...)` 把 DiT 上下文特征 F_ctx **丢弃**（死参数，梯度 None，占 4.76%）。改为 `context = context_cnn + F_ctx`，让 DiT 几何多一条直达 GRU 的通路（GRU input_dim 不变，ckpt 兼容）。

### 审计方法
9 个 agent 多视角扫 bug + 对抗性反驳（workflow `flow-corr-rootcause-audit`）：CorrBlock 根因被 3 个独立 lens 各自撞到、对抗验证全部"反驳失败"；被驳回 1 条（InstanceNorm 抹尺度——实验证伪，因 CorrBlock 前有 F.normalize，匹配信号本就尺度无关）。

### 自检脚本
[verify_corrblock_fix.py](verify_corrblock_fix.py)：开训前最后一道闸，4 项全绿。

### ⚠️ 重训要求
旧 ckpt（20260608/20260603 全部）都是在断掉的相关链路下学的，**权重无意义，必须从头重训**。修复后旧 ckpt 不可续训。
