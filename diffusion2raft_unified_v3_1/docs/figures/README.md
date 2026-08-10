# v3.3 unified-model overview

The overview separates the inference graph from training-only supervision and
uses the following visual convention:

- blue/purple: frozen external Qwen and geometry-teacher backbones;
- orange: trainable modules saved in the unified checkpoint;
- violet: deterministic backward-flow geometry;
- green: final flow-preserving image output;
- red dashed paths: training-only targets and losses.

Artifacts:

- `v33_unified_model_overview.svg`: editable vector master for papers and slides;
- `v33_unified_model_overview.pdf`: vector export for paper composition;
- `v33_unified_model_overview.png`: 2700×1500 raster preview.

Regenerate all three formats with:

```bash
cd /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/diffusion2raft_unified_v3_1
/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/bin/python \
  scripts/render_v33_architecture_overview.py
```

Suggested paper caption:

> **Overview of Diffusion2RAFT v3.3.** Frozen Qwen-Image-Edit supplies target
> and source diffusion features, while a frozen strong optical-flow teacher
> supplies the coarse backward-flow prior. Trainable projections, reliability
> fusion, and a RAFT-like recurrent refiner predict a bounded residual flow.
> The residual is geometrically composed—rather than directly added—with the
> prior to produce the final backward flow. Final RGB pixels are sampled from
> the original warped document. Solid paths are used during inference; dashed
> paths denote training-only supervision.

This is the architecture specified by the current v3.3 code. It must not be
presented as an already measured v3.3 result until the capacity check, formal
8-GPU smoke, training, and final quality evaluation have completed.

## Real C4 route-failure comparison

The sample-specific C4 comparison is intentionally generated under `runs/`
rather than committed here, because it contains actual output from the 3.5 GB
frozen teacher. Run:

```bash
bash /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp/slurm/v33_pipeline/visualize_teacher_399999_c4_route_failure_1gpu.sh
```

The resulting `c4_route_comparison.png` contrasts the recorded wrong `q=0°`
route with the teacher-optimal `q=90°` route for `Pers_NoAug_0010947`. Teacher
panels are actual model outputs. The 40 px-capped panels are GT-derived oracle
projections for explanation only, not learned-model predictions.
