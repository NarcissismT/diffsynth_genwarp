# V4.3 — 大搜索范围 + 多迭代 — 2026-06-02

## 一句话

在 V4.2 (path B 双图跨段 Q/K) 基础上，针对**过拟合诊断**精确定位的瓶颈做两处改动：
**CorrBlock radius 4→8 + iters 4→12**，直击"大位移搜索范围不足"。

## 决策依据：过拟合诊断结论

在 18 个固定分层样本上训 3000 步（`overfit_diagnose.py`）：

| 层 | zero EPE | 过拟合后 <5px | 判定 |
|----|---------|--------------|------|
| low (位移<22px) | 7.69 | **97.4%** | ✅ 几乎完美 |
| mid (22-32px) | 13.12 | **96.3%** | ✅ 几乎完美 |
| high (>32px) | 31.79 | **53.0%** | ❌ 卡住 |

**结论：不是容量瓶颈（低/中位移能完美拟合，证明 FlowHead 容量+DiT 特征质量都够），而是【大位移搜索范围不足】。**

V4.2 的 `CorrBlock(radius=4)` 在 H/8 分辨率搜索 ±4 → 等效原图 **±32px**。high 样本位移 31~169px **超出搜索范围** → correlation 找不到对应 → 只能靠 GRU 外推 → 精度差。

## V4.3 改动

| 项 | V4.2 | V4.3 | 效果 |
|----|------|------|------|
| CorrBlock radius | 4（±32px）| **8（±64px）** | 覆盖更多 high 位移 |
| corr 通道 (2r+1)² | 81 | **289** | ConvGRU input_dim 81→289 |
| iters（训练+推理）| 4 | **12** | RAFT 标准，GRU 多步挪大位移 |
| 总参数 | 2.07M | **2.61M** | radius 增大致 ConvGRU 变大 |

其余完全继承 V4.2：双图拼接、跨段 Q/K（Q←corrected/K←warped）、timestep/1000、`--k_source` 开关、grad loss、InstanceNorm。

**注意：radius 改变 → ConvGRU input_dim 改变，V4.2 ckpt(radius=4) 不能加载到 V4.3，需从头训练。**

## 文件结构

```
20260602-v4_3-bigcorr/
├── README.md
├── train_flow_head_v4_3.py        # 训练（radius=8, iters=12）
├── qwen_image_flow_v4_3.py        # 推理（flow_iters 默认 12）
├── overfit_diagnose.py            # 过拟合诊断（iters 默认 12）
├── overfit_samples.csv            # 18 个固定分层样本
├── utils/
│   └── flow_head_v4_3.py          # FlowHeadV4_3（radius=8, iters=12 默认）
└── scripts/
    ├── train_v4_3_layer36.sh      # 训练启动
    ├── flow_v4_3_sample.sh        # 推理启动
    └── overfit_diagnose.sh        # 诊断启动
```

## 怎么跑

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp

# 训练（从头，因为 radius 变了）
bash versions/20260602-v4_3-bigcorr/scripts/train_v4_3_layer36.sh

# 训练前先用诊断验证 V4.3 能否把 high 层过拟合上去（强烈推荐，半天出结论）
bash versions/20260602-v4_3-bigcorr/scripts/overfit_diagnose.sh
```

## 预期 & 验证方式

**预期**：high 层 <5px 从 53% → 70%+，整体从 82% → 88%+。

**最快验证（推荐先做）**：直接跑 `overfit_diagnose.sh`（V4.3 版）。如果 high 层 <5px 从 53% 明显抬升（比如 >75%），证明 radius+iters 改动确实解决了搜索范围瓶颈，再投入全量训练。如果 high 层仍卡 53% 左右，说明问题不在搜索范围（可能在 DiT 特征对大位移的表达能力），需重新诊断。

ckpt 输出：`/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/flow_head_v4_3_ckpts/`

## 训练成本提醒

iters 4→12 + radius 4→8 会让每步训练**显存和时间约增 2-3×**（correlation 计算量 289/81≈3.5×，GRU 迭代 12/4=3×）。若显存吃紧，可先验证诊断（单卡小集），或把 radius 退到 6（±48px，corr 通道 169）折中。
