# v3.3 Slurm production pipeline

这些文件是 Slurm **前台 job body**，固定绑定已完成的 v3.1
`epoch_0020.pt`（SHA-256
`7f5f743de8994907b916182fd3d1c1e81c0015b5b53fb8bf5038ff4c2ad17fe5`）。
不要使用任何 `run_*_background.sh`；Slurm 必须直接跟踪脚本退出状态。

运行镜像：

```text
docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers
```

## 当前状态：v6 清除 C4 catastrophe，先审计 24×40 结构安全

2026-07-28 的 Slurm job `78888` 已完整跑完 300 original、300 rotation 和
300 full-geometry 样本。环境与原始朝向均正常，但 `259999_raft_unwarp.pt` 在约
`|rotation| > 45°` 后出现 90° 朝向混淆：rotation/full-geometry 的 trainable coverage
分别只有 `0.246854`/`0.256268`，远低于 production 阈值 `0.985`。因此没有生成
`approved.json`，现在不要运行 `02_teacher_smoke_8gpu.sh` 或
`03_teacher_train_8gpu.sh`；原样重跑 `01` 只会得到同一个确定性拒绝。

2026-07-29 的 Slurm job `78920` 又完成了后期 `399999_raft_unwarp.pt` 的同规模
A/B。它虽然把 original EPE 从 `1.01194` 改善到 `0.86583`，但 rotation EPE
仍为 `218.50004`，rotation/full-geometry trainable coverage 仍只有
`0.247155`/`0.257254`。这与 259999 是同一个 90° 朝向混淆模式，因此不再完整扫描
中间的 299999/394999 checkpoint。

2026-07-29 的 Slurm job `78932` 已运行
`diagnostic_teacher_399999_oracle_c4_1gpu.sh`：利用合成样本已知 rotation 选择理想
quarter-turn 后，399999 的 rotation EPE 从 `218.5000` 降至 `2.1680`，降幅 `99.01%`；
四个 quarter-turn 的 EPE 都约为 `1.75–2.39 px`。这证明 teacher 的主要问题是离散 C4
朝向混淆。

该 v1 诊断先把 teacher flow 映回原 source frame，再运行现有六步 fixed-point residual
反解，因此 trainable coverage 只有 `32.30%`。按 quarter-turn 分组，0° 为 `96.72%`，
±90° 为 `14.11%/17.32%`，180° 仅 `0.0199%`；这是旋转 Jacobian 使 fixed-point 迭代
不再收缩，并非 teacher EPE 或 24 px cap 失败。

2026-07-29 的 Slurm job `78933` 随后完成 canonical-frame v2。它保持 teacher、GT absolute
map、residual target 和 fixed-point solver 都在 canonical source frame，只将最终 composed
absolute map 映回作为对照。300 个 rotation-isolated 样本的 canonical EPE 为 `2.16799 px`，
solver coverage 为 `0.983346`，trainable coverage 为 `0.980576`，overflow pixel rate 为
`0.002817`；四个 quarter-turn 的 solver/trainable coverage 均超过首轮研究阈值 `0.95`，
EPE 均约为 `1.75–2.39 px`。因此 canonical-frame 假设已通过研究门槛。

Slurm job `78943` 已完成后续 `6/12/24` 步 solver sweep。12 步是第一个达到 aggregate
solver/trainable 目标的设置：`0.995784`/`0.992237`，overflow 为 `0.003562`，stride
trainable EPE 为 `0.07034 px`；24 步只再改善约 0.1 个百分点，因此下一阶段固定 12 步。
原 rotation `30–60°` 的 solver/trainable coverage 已从 `0.931263`/`0.921874` 提升到
`0.990314`/`0.976054`。剩余约 `0.0144` overflow 是 24 px residual cap 问题，继续增加
solver 迭代无法消除。

Slurm job `78946` 已完成上述 conditional full-geometry oracle：solver coverage
`0.993225` 通过，但 overflow `0.008614` 高于 `0.005`，trainable coverage `0.984670`
也比 `0.985` 少 `0.000330`。原 rotation `30–60°` 与 canonical residual `40–45°` 的
solver coverage 分别只有 `0.971389`/`0.955586`，而 `90–120°` rotation overflow 为
`0.020304`。因此必须同时拆分 solver 迭代不足和 24 px cap 不足。

Slurm job `78953` 已完成 full-geometry capacity grid v5。`12×32` 是第一个通过 aggregate
四门槛的 cell：solver/overflow/trainable/stride EPE 为
`0.993225/0.003416/0.989832/0.069907`；`24×40` aggregate 也通过，分别为
`0.997099/0.003685/0.993425/0.063114`。但没有任何 cell 同时清除所有严格弱桶。
该已完成 job body 是
`diagnostic_teacher_399999_oracle_c4_canonical_full_geometry_capacity_grid_v5_1gpu.sh`，
报告保存在
`runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_capacity_grid_v5/candidate-399999.json`。
`24×40` 下，原 rotation `30–60°` 与 nearest canonical residual `40–45°` 的 overflow
仍为 `0.017918/0.029006`，trainable 只有 `0.976258/0.963453`。

Slurm job `78979` 已完成 all-four-C4 v6。nearest-angle 与最小 teacher-EPE 路由
`299/300` 一致；唯一错例正是 `Pers_NoAug_0010947`。相邻 `q=90°` 把该样本 teacher EPE
从 `250.277253` 降到 `9.871167 px`，24×40 overflow 从 `0.984449` 降到 `0`，trainable
coverage 提升到 `0.993790`。全体 best-EPE top1/top2 margin 最小为 `206.016 px`；
best-EPE 与 capacity-aware 标签在 300 个样本和六个 cell 上完全一致。

### 可选：生成唯一 C4 错例的真实对比图

下面的只读 job 不改变任何门禁状态。它重放 `Pers_NoAug_0010947`，对错误 `q=0°` 和正确
`q=90°` 各执行一次真实 399999 teacher forward，并生成一张适合直接查看或放进汇报的对比图：

```bash
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/slurm/v33_pipeline/visualize_teacher_399999_c4_route_failure_1gpu.sh
```

资源为 1 GPU、16 CPU，建议 30 分钟。输出目录是
`runs/v33_diagnostics/teacher_c4_route_failure_visualization_v1/job-$SLURM_JOB_ID/`，包含
`c4_route_comparison.png`、八张单独面板和 `report.json`；完成标志为
`D2R_V33_C4_ROUTE_VISUALIZATION_COMPLETE`。其中 teacher 图像是真实输出，40 px-cap 图像是
GT residual 构造的解释性 oracle 投影，不是训练模型输出，也不会写 `approved.json`。

best-EPE `24×40` aggregate 的 solver/overflow/trainable/stride EPE 为
`0.997096/0.000736/0.996363/0.063028`。同时对 aggregate、7 个原 rotation 桶和 4 个
nearest-residual 桶执行冻结门槛后，只有该 cell 零失败；`12×40` 仍有 3 项
solver/trainable 失败，`24×32` 仍有 2 项 overflow 失败。不能只改 solver 或 cap，也不能
据 aggregate 直接采用 `12×32`。

当前下一步是 v7 结构安全上界：以 SHA-256 固定绑定 v6 报告，逐样本回放 best-EPE C4
标签，并检查 24×40 的 stride-8 residual 是否引入 in-bounds、fold/Jacobian、线条、曲率
或纹理风险。

```bash
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/slurm/v33_pipeline/diagnostic_teacher_399999_best_of_c4_structural_safety_v7_1gpu.sh
```

该作业需要 1 GPU、16 CPU、建议 2 小时；固定使用 SHA-256 为
`e27c12a7085364b95773304d6a879567b5579134889553f2c4d39f02f6263fc5` 的 399999 teacher。
绑定的 v6 报告 SHA-256 为
`abf55cc22ea65665d175563c73d18ce993d7401923c4afe4e824073900655176`。报告写到
`runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_structural_safety_v7/candidate-399999.json`，
完成标志为 `D2R_V33_BEST_OF_C4_STRUCTURAL_SAFETY_COMPLETE`。

v7 对 24/32/40 px 三个 support 复算 capacity，但冻结选择为 24 次 solver、40 px cap，
并以 24 px 为新纳入 support 的基线。它逐样本复验 v6 的 index/seed/homography、C4 标签和
capacity 数值；每个样本只运行一次选中 teacher。结构门槛沿用 typical-v3.3 的 fold/Jacobian
阈值，并加入 line/curvature 不劣于 teacher、flow EPE `<=1 px`、纹理相对 teacher 至少恢复
90%（允许 `1/255` 插值 floor）的冻结检查。已完成的 v6 报告保存在
`runs/v33_diagnostics/teacher_quarter_turn_oracle_canonical_full_geometry_best_of_c4_v6/candidate-399999.json`。

v6/v7 都使用部署时不可获得的 GT flow，报告 kind 与 production capacity 完全不同；即使
v7 `passed=True`，也不会写 `approved.json`、receipt 或解锁 `02`/`03`。它只允许下一步实现
只读 warped 图像、带 margin/拒识路径的 deployable router；随后必须用真实 router 选择重跑
同一 capacity 和结构门禁。

以下顺序只供未来 oracle、full-geometry 和真实 router 全部门禁通过并正式解冻后使用；当前
不要提交其中任何一个作业。解冻后两个 8 GPU 作业都必须是**同一台节点上的 8 张可见 GPU**：

1. `01_teacher_capacity_1gpu.sh`：1 GPU、16 CPU、建议 12 小时。
2. `02_teacher_smoke_8gpu.sh`：8 GPU、至少 40 CPU、建议 3 小时。
3. `03_teacher_train_8gpu.sh`：8 GPU、至少 40 CPU；从 completed 20 训练到 32，建议约 120 小时。

推荐在 Slurm 平台的 job command 中直接调用共享存储上的绝对路径：

```bash
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/slurm/v33_pipeline/01_teacher_capacity_1gpu.sh
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/slurm/v33_pipeline/02_teacher_smoke_8gpu.sh
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/slurm/v33_pipeline/03_teacher_train_8gpu.sh
```

如果平台会先把“上传的脚本”复制到临时目录，不要只上传单个阶段文件，因为它需要同目录
的 `common.sh`。此时可以上传一个仅含上述单行绝对路径命令的 wrapper；或者把所需 job
body 与 `common.sh` 保持相对位置一起上传。

如果使用原生 `sbatch`，请在集群自己的 batch header 中补充 partition、account、
container 和日志设置，然后用 `afterok` 串联。仓库没有臆造这些集群专属选项。例如三个
batch header 分别调用上述三个绝对路径后，可提交：

```bash
capacity_job=$(sbatch --parsable 01_capacity.sbatch)
smoke_job=$(sbatch --parsable --dependency=afterok:"$capacity_job" 02_smoke.sbatch)
sbatch --dependency=afterok:"$smoke_job" 03_train.sbatch
```

单卡作业保留 Slurm 设置的 `CUDA_VISIBLE_DEVICES`，并严格要求 PyTorch 只看到一张卡；
这张唯一可见卡在进程内自然是 teacher 所需的逻辑 `cuda:0`。

每阶段的权威成功标志：

- capacity：`D2R_V33_CAPACITY_PASS`，并生成
  `runs/preflight_v33_teacher_capacity/approved.json`；
- smoke：`D2R_V33_FORMAL_SMOKE_JOB_PASS`，且固定报告
  `runs/v33_smoke_reports/epoch0020_formal/overall_report.json` 通过深检；
- train：`D2R_V33_TRAIN_COMPLETE`，且 `epoch_0032.pt`、`latest.pt`、`anchor.pt`
  和 anchor/final capacity receipt 全部通过深检。

正式 smoke 的输出目录采用固定名称并拒绝覆盖。若 smoke 失败后需要重跑，请先把
`runs/v33_smoke_reports/epoch0020_formal` 改名保存，以免混用失败证据。v3.3 训练则可在
Slurm TIMEOUT 后原样重投 `03_teacher_train_8gpu.sh`，launcher 会优先恢复 v3.3 自己的
`latest.pt`。

这里的脚本有意操作 workspace 权威 run tree；data 目录是同步代码镜像，不保存
workspace 的 `runs/`、日志或 checkpoint。
