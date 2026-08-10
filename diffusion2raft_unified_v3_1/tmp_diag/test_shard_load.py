import time, torch
t0 = time.time()
from diffusers import QwenImageEditPipeline
from diffusers import QwenImageTransformer2DModel
model_id = "/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"

n_gpu = torch.cuda.device_count()
print("visible GPUs:", n_gpu)
# Reserve GPU0 for the rectifier + projections + text_encoder; shard transformer on GPU1..
max_memory = {i: "20GiB" for i in range(n_gpu)}

print("[1] loading sharded transformer via device_map=auto ...", flush=True)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    device_map="auto", max_memory=max_memory, local_files_only=True,
)
print(f"    transformer loaded in {time.time()-t0:.1f}s", flush=True)
try:
    dm = transformer.hf_device_map
    from collections import Counter
    print("    device_map spread:", Counter(str(v) for v in dm.values()))
except Exception as e:
    print("    (no hf_device_map)", e)

print("[2] loading pipeline with sharded transformer ...", flush=True)
pipe = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True,
)
print(f"    pipeline built in {time.time()-t0:.1f}s", flush=True)
# Put text_encoder + vae on a lightly-loaded GPU (last one)
te_dev = f"cuda:{n_gpu-1}"
pipe.text_encoder.to(te_dev)
pipe.vae.to(te_dev)
print(f"    text_encoder+vae on {te_dev}", flush=True)
for i in range(n_gpu):
    free, tot = torch.cuda.mem_get_info(i)
    used = (tot-free)/1e9
    if used > 0.2:
        print(f"    GPU{i}: used {used:.1f} GB / {tot/1e9:.1f} GB")
print("LOAD_OK")
