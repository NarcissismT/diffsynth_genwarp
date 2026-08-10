import json, os, collections, random
import numpy as np

train = "data/train.jsonl"
cats = collections.Counter()
recs = []
with open(train) as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        cats[r.get("category","?")] += 1
        recs.append(r)
print("total:", len(recs))
for c, n in cats.most_common():
    print(f"  {c:<24} {n}")

# sample some GT flows: estimate global rotation content of the backward flow
random.seed(0)
sample = random.sample(recs, min(30, len(recs)))
rot_deg = []
mag = []
for r in sample:
    fp = r["flow"]
    if not os.path.isabs(fp): fp = os.path.join(os.path.dirname(train), fp)
    try:
        fl = np.load(fp)
    except Exception:
        continue
    if fl.shape[-1] != 2: fl = np.moveaxis(fl, 0, -1)
    H, W, _ = fl.shape
    # backward flow: source = grid + flow. Fit affine source ~ A*[x,y]+b to get rotation.
    ys, xs = np.mgrid[0:H, 0:W]
    gx = xs.ravel().astype(np.float64); gy = ys.ravel().astype(np.float64)
    sx = (gx + fl[...,0].ravel()); sy = (gy + fl[...,1].ravel())
    finite = np.isfinite(sx) & np.isfinite(sy)
    # subsample
    idx = np.random.choice(np.where(finite)[0], size=min(20000, finite.sum()), replace=False)
    A = np.stack([gx[idx], gy[idx], np.ones_like(gx[idx])], 1)
    # solve for x-map and y-map
    cx, *_ = np.linalg.lstsq(A, sx[idx], rcond=None)
    cy, *_ = np.linalg.lstsq(A, sy[idx], rcond=None)
    # rotation angle from the linear part [[cx0,cx1],[cy0,cy1]]
    ang = np.degrees(np.arctan2(cy[0]-cx[1], cx[0]+cy[1]))  # rough
    rot_deg.append(abs(ang))
    mag.append(float(np.nanmean(np.hypot(fl[...,0], fl[...,1]))))
rot_deg = np.array(rot_deg); mag = np.array(mag)
print(f"\nsampled {len(rot_deg)} flows")
print(f"abs global rotation deg: mean={rot_deg.mean():.2f} p50={np.median(rot_deg):.2f} p90={np.percentile(rot_deg,90):.2f} max={rot_deg.max():.2f}")
print(f"mean flow magnitude px (on native GT canvas): mean={mag.mean():.1f} p90={np.percentile(mag,90):.1f}")
