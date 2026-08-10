# DocGrid-Flow v2 architecture and code index

This document is the canonical implementation map for
`Diffusion2RAFT_Plan_and_Goals_newest.md`.

## Input and supervision contract

Inference has exactly one model input: the warped RGB image `Iw`.

Training passes `Mgt` (the GT absolute backward map) and `valid_mask` only to
construct losses and the residual Flow-Matching target. The rectified image is
an auxiliary supervised target. Neither `Mgt` nor rectified RGB is concatenated
to the image encoder or available at inference.

```text
inference: Iw -----------------------------------------------> model
training:  Iw -> model;  Mgt + valid_mask + rectified RGB ---> targets/losses
```

Stage-5 structure patches do not crop the source condition. `Iw` remains the
complete warped page, while `target_window=[x0,y0,w,h]` selects a window in the
canonical target canvas. Every predicted/GT map value remains an absolute
coordinate in the complete source image. This makes patch training and
full-page inference share one map convention.

## Complete forward path

```text
warped RGB Iw
  |-- CNN/FPN ---------------------------------------------- local4, fpn8
  |-- H/V structure encoder ------------------------------- hv4, hv8, H/V/B logits
  `-- Qwen-Image-Edit VAE/condition preparation + frozen DiT probe
       `-- selected hidden or Q/K features -> DPT/FPN adapter
                    |
                    v
       spatial softmax gated fusion(Qwen, CNN, H/V) at 1/8
                    |
          coarse map head + confidence head
                    |
     Mc (absolute backward map) + C (confidence)
                    |
     residual Coordinate Flow Transformer (8 blocks)
       token = [Rt, Mc-P, P, C]
       coordinate self-attention
       cross-attention to fused visual features
       AdaLN/time conditioning + confidence-gated residual
       H/V-gated feed-forward
                    |
       ODE residual proposal Rhat
                    |
       Md = Mc + G(C) * Rhat
                    |
       resize absolute map to 1/4
                    |
       WARR ConvGRU x 4
       geometry cues: displacement, first/second derivatives,
       mixed derivative, determinant; plus warped CNN/HV features
                    |
       learned bounded update gate and delta map
                    |
       RAFT-style convex map upsampling
                    |
       full-resolution absolute backward map Mfinal
                    |
       one grid_sample(original native-HD Iw, Mfinal)
                    |
       rectified RGB
```

Qwen's VAE decoder is never used. The requested Qwen pipeline output is a
latent placeholder that is discarded after feature hooks run. The only final
RGB decoder/renderer is the single `grid_sample` from the original warped
image; therefore the path cannot invent or rewrite characters by decoding new
pixels.

## Residual Flow-Matching contract

At the 1/8 grid:

```text
R*       = Mgt - Mc
G(C)     = g_min + (1 - g_min) * (1 - C)
proposal = clip(R* / G(C))
sigma(C) = sigma_min + (sigma_max - sigma_min) * (1 - C)
R0       = sigma(C) * epsilon
Rt       = (1-t)R0 + t*proposal
v*       = proposal - R0
Md       = Mc + G(C) * Rhat
```

Confidence is detached for target/noise construction by default. Public map
sequences are absolute maps in source-pixel x/y coordinates; residual states
are exposed separately and cannot be confused with map supervision.

## Canonical files

| Responsibility | Canonical file | Compatibility name |
|---|---|---|
| Unified model and three-way fusion | `models/docgrid_flow.py` | `models/full.py` |
| Residual Coordinate Flow Transformer | `models/coordinate_flow_transformer.py` | `models/coordinate_flow.py` |
| WARR and convex upsampler | `models/warr.py` | `models/refiner.py` |
| Frozen Qwen probe and DPT/FPN adapter | `models/qwen_feature_probe.py` | `models/qwen_condition.py` |
| Real-Qwen runtime validator | `validate_qwen_runtime.py` | — |
| Geometry coordinate transforms | `geometry.py` | — |
| Full losses | `losses.py` | — |
| Stage-0 payload audit/frozen contract | `audit_data.py` | — |
| Exact analytic GT renderer | `render_analytic_gt.py` | — |
| Analytic shard contract merger | `merge_analytic_shards.py` | — |
| Legacy RAFT pseudo migration | `migrate_legacy_raft.py` | — |
| Stage-5 three-view sampler | `training_views.py` | — |
| Shared stage-aware trainer | `train_full.py` | — |
| Stage-specific training entry points | `training/train_*.py` | — |
| Native single-sample inference | `infer_full.py` | — |
| Exact-pixel evaluation | `evaluation/evaluator.py` | `evaluate_full.py` |
| Gate 1-5 hard criteria/receipts | `gates.py` | — |
| Three-seed aggregation | `aggregate_seeds.py` | — |
| Repeatability verification | `verify_repeatability.py` | — |
| Runtime ablation matrix | `ablation_suite.py` | — |
| OCR CER/WER evidence | `evaluation/ocr_metrics.py` | — |

Data provenance, immutable asset identities and exact preparation commands are
centralized in `docs/DATA_PROVENANCE.md`.

Compatibility modules contain imports only and must not receive new logic.
The Python package name `cp_docflow` is retained so old commands/checkpoints
fail explicitly instead of breaking imports silently; all new architecture,
checkpoint, config, stage, and artifact names use DocGrid-Flow v2.

## Stage topology

| Stage | Runtime path | Trainable modules |
|---|---|---|
| 1 `coarse` | CNN/fusion -> coarse -> basic convex | CNN/FPN, coarse/confidence, fusion projection, convex; no Qwen adapter state |
| 2 `warr` | coarse -> H/V -> WARR -> convex | first 20% H/V/WARR/convex; then small-LR coarse/fusion joint tuning; no Qwen adapter state |
| 3 `coord_fm` | coarse -> CFT -> WARR -> convex | first 70% CFT/tokenizer/velocity; then small-LR deterministic/refiner joint tuning; no Qwen adapter state |
| 4 `qwen` | all paths with frozen Qwen | newly initialized DPT/FPN adapter and fusion only |
| 5 `full_page` | complete model | all except frozen Qwen source |

The old strings `refiner` and `joint` are accepted only as aliases for `warr`
and `full_page` when reading legacy code.

## Reproducibility and evaluation path

```text
Stage-0 audit
  -> frozen_contract.json (manifest + every payload SHA + split + seeds)
  -> stage training (expanded config + run_manifest + checkpoint training_seed)
  -> gate evaluation (no runtime override)
  -> metrics/per_sample/calibration/runtime/fixed visual artifacts
  -> Gate-5 model/oracle/target PNG export + SHA-bound ocr_images.jsonl
  -> OCR transcripts bound to exact image bytes/sample IDs/checkpoint/manifest
  -> hard Gate criteria + human review evidence
  -> immutable receipt
  -> next stage preflight
```

Exploratory ablation evaluation is deliberately separate from gate evaluation.
A report with runtime overrides is never gate-eligible. Final claims require at
least three distinct checkpoint seeds and a repeatability result. Gate 5 also
rejects manually copied OCR values unless the immutable OCR report is bound to
the same geometry evaluation identities, dataset payload, image-manifest SHA,
and per-row model/oracle image SHA values.
