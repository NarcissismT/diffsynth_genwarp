import os
from PIL import Image

warp_dir = "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical"
base_dir = "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/test_silver_bullet_imgs/typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999"
names = sorted(os.listdir(warp_dir))[:3]
os.makedirs("tmp_diag/peek", exist_ok=True)
for n in names:
    w = Image.open(os.path.join(warp_dir, n)).convert("RGB")
    bpath = os.path.join(base_dir, n)
    b = Image.open(bpath).convert("RGB") if os.path.exists(bpath) else None
    print(n, "warped", w.size, "baseline", (b.size if b else None))
    # side by side, scaled to height 512
    def scale(im):
        h = 512; ww = int(im.width * h / im.height); return im.resize((ww, h))
    ws = scale(w); parts = [ws]
    if b: parts.append(scale(b))
    total_w = sum(p.width for p in parts) + 10*(len(parts)-1)
    canvas = Image.new("RGB", (total_w, 512), "white")
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0)); x += p.width + 10
    canvas.save(f"tmp_diag/peek/{n}_warp_vs_base.jpg", quality=90)
print("done")
