#!/usr/bin/env python3
"""
划分 held-out 验证集（按难度分层）— V4.4 验证集 step 1/3
=========================================================
从 metadata_with_vae.csv 切出一个【从不参与训练】的验证集，并按样本难度
（flow_gt 的 zero-flow EPE = 平均位移模长）分 low/mid/high 三层均衡抽样，
否则随机抽样会偏向简单样本，测不出难样本泛化（你真正关心的）。

输出：
  metadata_train.csv  — 训练用（= 全量 - 验证集）。训练脚本 csv 换成这个。
  metadata_val.csv    — 验证用（含 zero_epe / difficulty 列，供分层报告）

难度分层逻辑：
  1. 从全量随机抽 num_probe 行作候选池（category 自然按比例覆盖）
  2. 读各自 flow_gt 算 zero_epe，按 33/66 分位切 low/mid/high
  3. 每层等量抽 num_val//3 行组成验证集

跑法（一次性，纯 CPU 读 npy，约几分钟）：
  cd versions/20260603-v4_4-hardweight
  /juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/envs/RAFT_flow/bin/python make_val_split.py \
      --csv /juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_vae.csv \
      --num_val 300 --num_probe 1500 --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def zero_flow_epe(npy_path: str) -> float:
    """读 flow_gt，算 zero-flow EPE（平均位移模长，单位 px）。读不到返回 -1。"""
    try:
        flow = np.load(npy_path)
    except Exception:
        return -1.0
    # 兼容 (2,H,W) 与 (H,W,2)
    if flow.ndim == 3 and flow.shape[0] == 2:
        mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    elif flow.ndim == 3 and flow.shape[-1] == 2:
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    else:
        return -1.0
    return float(mag.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="全量 metadata csv")
    ap.add_argument("--num_val", type=int, default=300, help="验证集总量（会被 3 整除分层）")
    ap.add_argument("--num_probe", type=int, default=1500,
                    help="候选池大小，从中算难度分层抽验证集。越大分层越准但读 npy 越慢")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_train", default=None)
    ap.add_argument("--out_val", default=None)
    args = ap.parse_args()

    base = os.path.dirname(args.csv)
    out_train = args.out_train or os.path.join(base, "metadata_train.csv")
    out_val = args.out_val or os.path.join(base, "metadata_val.csv")

    df = pd.read_csv(args.csv)
    print(f"全量 {len(df)} 行，category 分布:\n{df['category'].value_counts()}\n")

    rng = np.random.RandomState(args.seed)

    # 1. 候选池（随机抽，category 自然按比例覆盖）
    probe_n = min(args.num_probe, len(df))
    probe_idx = rng.choice(len(df), size=probe_n, replace=False)
    print(f"候选池 {probe_n} 行，读 flow_gt 算 zero-flow EPE（难度）...")

    epes = []
    valid_idx = []
    for i in tqdm(probe_idx):
        epe = zero_flow_epe(df.iloc[i]["flow_gt_path"])
        if epe >= 0:
            epes.append(epe)
            valid_idx.append(i)
    epes = np.array(epes)
    valid_idx = np.array(valid_idx)
    print(f"有效候选 {len(valid_idx)} 行，EPE 范围 {epes.min():.1f}~{epes.max():.1f}px，"
          f"中位数 {np.median(epes):.1f}px")

    # 2. 按 33/66 分位分层
    q33, q66 = np.percentile(epes, [33, 66])
    print(f"难度分层阈值: low<{q33:.1f}px ≤ mid <{q66:.1f}px ≤ high")
    layers = {
        "low":  valid_idx[epes < q33],
        "mid":  valid_idx[(epes >= q33) & (epes < q66)],
        "high": valid_idx[epes >= q66],
    }
    epe_map = {valid_idx[k]: epes[k] for k in range(len(valid_idx))}

    # 3. 每层等量抽
    per = args.num_val // 3
    chosen = []
    for name, idxs in layers.items():
        take = min(per, len(idxs))
        pick = rng.choice(idxs, size=take, replace=False)
        chosen.extend(pick.tolist())
        print(f"  {name}: 候选 {len(idxs)} → 抽 {take}")
    chosen = sorted(set(chosen))

    # 4. 输出 val（附难度列）+ train（去掉 val）
    val_df = df.iloc[chosen].copy()
    val_df["zero_epe"] = [epe_map[i] for i in chosen]
    val_df["difficulty"] = pd.cut(
        val_df["zero_epe"], bins=[-1, q33, q66, 1e9],
        labels=["low", "mid", "high"])
    val_df.to_csv(out_val, index=False)

    train_df = df.drop(index=chosen).reset_index(drop=True)
    train_df.to_csv(out_train, index=False)

    print(f"\n✅ 验证集 {len(val_df)} 行 → {out_val}")
    print(f"   分层: {val_df['difficulty'].value_counts().to_dict()}")
    print(f"✅ 训练集 {len(train_df)} 行 → {out_train}（已剔除验证样本）")
    print(f"\n下一步: 训练脚本 csv_path 改成 {out_train}；"
          f"再跑 gen_val_corrected_low.py 预生成验证集的 corrected_low")


if __name__ == "__main__":
    main()
