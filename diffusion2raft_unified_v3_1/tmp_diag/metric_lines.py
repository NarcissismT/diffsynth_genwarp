"""No-reference dewarp quality proxy for document photos.

For a well-rectified page, dominant straight lines (table rules, text rows,
page edges) should be axis-aligned and straight.  We detect line segments with
OpenCV and measure:
  - orientation error: how far each segment's angle is from the nearest of
    {0, 90} degrees (deg). Lower = more axis-aligned.
  - fraction of near-axis segments (|angle to nearest axis| < 5 deg).
Aggregated over all detected segments, weighted by segment length.
"""
import os, sys, json, math
import numpy as np
import cv2


def line_metrics(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    scale = 1600.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(img)[0]
    if lines is None or len(lines) == 0:
        return dict(n=0, orient_err_deg=float('nan'), axis_frac=float('nan'))
    lines = lines.reshape(-1, 4)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))  # -180..180
    a = np.abs(ang) % 90.0
    dist_to_axis = np.minimum(a, 90.0 - a)  # 0..45, distance to nearest of 0/90
    # only consider reasonably long segments
    keep = length > (0.03 * max(img.shape))
    if keep.sum() == 0:
        keep = length > 0
    length = length[keep]; dist_to_axis = dist_to_axis[keep]
    w_sum = length.sum()
    orient_err = float((dist_to_axis * length).sum() / max(w_sum, 1e-6))
    axis_frac = float((length[dist_to_axis < 5.0].sum()) / max(w_sum, 1e-6))
    return dict(n=int(keep.sum()), orient_err_deg=orient_err, axis_frac=axis_frac)


def main():
    warp_dir = "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical"
    base_dir = "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/test_silver_bullet_imgs/typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999"
    uni_dir = "runs/d2r_v3_1/typical_unified_letterbox"
    names = sorted(n for n in os.listdir(warp_dir) if n.lower().endswith('.jpg'))
    agg = {k: [] for k in ("warped", "prior", "unified", "baseline")}
    per = []
    for name in names:
        stem = name[:-4]
        srcs = {
            "warped": os.path.join(warp_dir, name),
            "prior": os.path.join(uni_dir, stem + "_prior_rectified.png"),
            "unified": os.path.join(uni_dir, stem + "_rectified.png"),
            "baseline": os.path.join(base_dir, name),
        }
        row = {"name": name}
        for k, p in srcs.items():
            m = line_metrics(p) if os.path.exists(p) else None
            if m:
                agg[k].append(m["orient_err_deg"])
                row[k] = round(m["orient_err_deg"], 3)
        per.append(row)
    print(f"{'method':<10}{'mean_orient_err_deg':>22}{'median':>10}")
    summary = {}
    for k in ("warped", "prior", "unified", "baseline"):
        v = np.array([x for x in agg[k] if not math.isnan(x)])
        summary[k] = dict(mean=float(v.mean()), median=float(np.median(v)), n=len(v))
        print(f"{k:<10}{v.mean():>22.3f}{np.median(v):>10.3f}")
    # win counts unified vs baseline (lower orient_err better)
    wins = sum(1 for r in per if 'unified' in r and 'baseline' in r and r['unified'] < r['baseline'])
    both = sum(1 for r in per if 'unified' in r and 'baseline' in r)
    print(f"\nunified beats baseline on orient_err: {wins}/{both}")
    up = sum(1 for r in per if 'unified' in r and 'prior' in r and r['unified'] < r['prior'])
    print(f"unified beats prior on orient_err:    {up}/{both}")
    with open("tmp_diag/line_metrics.json", "w") as f:
        json.dump({"summary": summary, "per_image": per}, f, indent=2)
    print("wrote tmp_diag/line_metrics.json")


if __name__ == "__main__":
    main()
