# DocGrid-Flow newest-plan requirement matrix

Updated 2026-07-31 against `Diffusion2RAFT_Plan_and_Goals_newest.md`.

This matrix separates **implemented engineering contracts** from **formal
experimental evidence**. “Implemented” means the production path and fail-closed
tests exist; it does not mean a real-data Gate has passed.

## Architecture and data path

| Newest-plan requirement | Status | Implementation/evidence |
|---|---|---|
| One warped RGB image is the only inference input | Implemented | `models/docgrid_flow.py`; GT map/mask/rectified RGB enter losses only |
| Qwen becomes a frozen geometry feature source | Implemented; real runtime pending | `models/qwen_feature_probe.py`, `validate_qwen_runtime.py`; real 20B receipt still needs A800 |
| No Qwen VAE Decoder in the output path | Implemented | Decoder guard plus `qwen_vae_decoder_used=false`; final RGB source is recorded |
| CNN/FPN and H/V structure paths | Implemented | Local high-resolution and horizontal/vertical/boundary feature paths |
| DPT/FPN alignment and gated three-way fusion | Implemented | Qwen/CNN/HV spatial gate with Stage-4-only adapter initialization |
| Coarse absolute backward map and confidence | Implemented | Source-pixel x/y map, `align_corners=False`, confidence/log-variance supervision |
| Confidence-protected residual coordinate flow | Implemented | Residual tokenizer, eight CFT blocks, confidence-dependent noise/gating, Euler integration |
| WARR recurrent refinement | Implemented | 1/4 feature warping, geometry/Jacobian cues, shared ConvGRU and bounded map updates |
| Convex upsampling | Implemented | RAFT-style learned convex upsampling with bilinear fallback tests |
| Exactly one final sample from native warped RGB | Implemented | One final `grid_sample`; no generative RGB decoder or intermediate RGB cascade |
| Full-page/patch coordinate consistency | Implemented | Target-window patches retain the complete source and absolute source coordinates |

## Training, losses and reproducibility

| Newest-plan requirement | Status | Implementation/evidence |
|---|---|---|
| Stage 0–5 training topology | Implemented | Stage-specific entry points/configs and parent/receipt checks |
| Stage 2 WARR-only then joint tuning | Implemented | First 20% refiner-only, final 80% small-LR joint tuning |
| Stage 3 velocity-only then joint tuning | Implemented | First 70% CFT/velocity, final 30% small-LR complete-chain tuning |
| Stage 4 frozen-Qwen fusion | Implemented; receipt pending | Adapter/fusion only; Qwen weights excluded from checkpoints |
| Stage 5 three-view full-page training | Implemented; corpus pending | Low-page, structure-patch and full-page sampler |
| Map/confidence/velocity/sequence/warp/structure losses | Implemented | Valid-mask-aware EPE, SSIM, gradient, H/V, bend, Jacobian, preserve and anti-fold losses |
| Immutable dataset identity and document separation | Implemented | Manifest plus every payload SHA, provenance, split leakage and fold audit |
| Fixed three seeds and repeatability | Tooling implemented; formal runs pending | Seeds 1337/2027/3407, aggregation and exact-pixel repeatability tools |

## Evaluation and acceptance

| Newest-plan requirement | Status | Implementation/evidence |
|---|---|---|
| EPE/P95/line/edge/straightness/fold metrics | Implemented | Evaluator schema v3 plus per-sample/group tables |
| Jacobian/bending/orthogonality/border metrics | Implemented | Geometry quality and page-boundary summaries |
| Image/text/table preservation metrics | Implemented | L1/PSNR/SSIM, character-edge preservation and table-line connectivity |
| OCR CER/WER tied to exact model/oracle images | Implemented; transcripts pending | Gate 5 exports PNGs and SHA-bound `ocr_images.jsonl`; OCR schema v2 verifies every image byte and geometry identity |
| Frozen supervised prior for Gate 1 | Implemented; full-corpus evaluation pending | Exact 259999 checkpoint adapter and immutable same-manifest evaluator-v3 baseline |
| Same-canvas deterministic baseline for Gate 5 | Implemented; run pending | Stage-2 WARR is re-evaluated on the 1024x768 full-page validation manifest |
| Gate 1–5 fail-closed receipts | Implemented | Hard thresholds plus baseline, seed, runtime, OCR and reviewed visual evidence |
| A1–A20 ablations | Partially prepared | Named runtime suite covers 15 engineering variants; formal A1–A20 training/evidence remains pending |
| Real-camera/full-resolution quality acceptance | Pending external data/runs | No formal Stage-5 metric, OCR or visual acceptance evidence exists yet |

## Current execution boundary

The next mandatory chain is:

```text
64-way analytic render
  -> merge
  -> frozen-prior evaluation on the merged validation manifest
  -> Stage-0 audit/freeze
  -> Stage 1 -> reviewed Gate 1 -> Stage 2 -> ... -> Stage 5 -> Gate 5
```

Real Qwen validation is independent of the first four jobs and may run in
parallel, but its immutable receipt is mandatory before Stage 4. The separate
1024x768 full-page render/merge/audit chain may run alongside Stages 1–4 and is
mandatory before Stage 5.

Therefore the model/training/evaluation implementation follows the newest-plan
logic, while the research goal remains open until the formal corpus, real Qwen
receipt, three-seed Stage 1–5 runs, A1–A20 evidence, OCR evidence and final
visual/metric Gates have actually passed.
