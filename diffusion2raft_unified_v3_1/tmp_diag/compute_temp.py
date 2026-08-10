import math, yaml
with open("configs/unified.yaml") as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
target = float(mc.get("correlation_temperature", 0.10))
start = float(mc.get("correlation_temperature_start", target))
start_epoch = int(mc.get("correlation_ramp_start_epoch", 1))
ramp_epochs = int(mc.get("correlation_ramp_epochs", 1))

def sched(epoch):
    display_epoch = int(epoch) + 1
    if display_epoch <= start_epoch:
        return start
    if ramp_epochs <= 1 or display_epoch >= start_epoch + ramp_epochs - 1:
        return target
    progress = (display_epoch - start_epoch) / max(ramp_epochs - 1, 1)
    return math.exp((1.0 - progress) * math.log(start) + progress * math.log(target))

print("target (infer/ablate default):", target, " start:", start,
      " ramp_start_epoch:", start_epoch, " ramp_epochs:", ramp_epochs)
for epoch in range(8, 14):
    print(f"  train epoch(0-idx)={epoch} display={epoch+1}: corr_t={sched(epoch):.4f}")

import subprocess
print("\n=== where does diffusion2raft resolve? ===")
