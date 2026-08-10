import os, sys
from PIL import Image, ImageDraw

warp_dir = "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical"
base_dir = "/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/test_silver_bullet_imgs/typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999"
# column dirs: label -> (dir, suffix)
cols = [("warped", warp_dir, None)]
for arg in sys.argv[1:]:
    label, d, suf = arg.split("::")
    cols.append((label, d, suf or None))
cols.append(("baseline", base_dir, None))

out_dir = os.environ.get("MONTAGE_OUT", "tmp_diag/montage")
os.makedirs(out_dir, exist_ok=True)
names = sorted(n for n in os.listdir(warp_dir) if n.lower().endswith(".jpg"))
H = 640
def load(d, name, suf):
    if suf:
        p = os.path.join(d, os.path.splitext(name)[0] + suf)
    else:
        p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    return Image.open(p).convert("RGB")
def scale(im):
    w = int(im.width * H / im.height); return im.resize((max(1,w), H))
count = 0
for name in names:
    imgs = []
    for label, d, suf in cols:
        im = load(d, name, suf)
        imgs.append((label, scale(im) if im else None))
    widths = [im.width if im else 200 for _, im in imgs]
    lab_h = 26
    total_w = sum(widths) + 8*(len(imgs)-1)
    canvas = Image.new("RGB", (total_w, H + lab_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (label, im), w in zip(imgs, widths):
        if im: canvas.paste(im, (x, lab_h))
        draw.text((x+4, 6), label, fill="black")
        x += w + 8
    canvas.save(os.path.join(out_dir, f"{os.path.splitext(name)[0]}.jpg"), quality=88)
    count += 1
print(f"wrote {count} montages to {out_dir}")
