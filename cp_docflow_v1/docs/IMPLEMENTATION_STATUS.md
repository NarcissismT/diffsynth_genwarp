# DocGrid-Flow v2 implementation status

Updated 2026-07-31 against `Diffusion2RAFT_Plan_and_Goals_newest.md`.

The plan-by-plan code/evidence split is tracked in
`docs/REQUIREMENT_MATRIX.md`.

## Engineering implementation complete

- One warped RGB image is the only inference condition. GT map, valid mask and
  rectified RGB are training targets only.
- The unified topology is CNN/FPN + H/V + frozen Qwen feature probe, three-way
  gated fusion, coarse map/confidence, residual Coordinate Flow Transformer,
  WARR ConvGRU, convex map upsampling, and one native-image `grid_sample`.
- Qwen VAE Decoder is excluded from the output path. Stage 1–3 checkpoints do
  not instantiate the untrained Qwen adapter; Stage 4 initializes it after the
  deterministic parent, and no checkpoint stores the 20B Qwen weights.
- The production hook validates the Qwen block tuple/token contract at runtime.
  A separate GPU receipt guards `vae.decode`, records selected target/source
  shapes, memory/parameter counts and local model config hashes; Stage 4/5
  preflight requires that receipt.
- Qwen receipt v2 also binds the complete formal probe contract (512x512,
  hidden layers -24/-12/-1, BF16, one deterministic step, CPU offload, latent
  output and 3072-channel target/source shapes). Validation overrides can no
  longer certify a Stage-4 topology that differs from the fixed model config.
- Absolute source-pixel x/y backward-map semantics, `align_corners=False`,
  window-aware resize/crop and native rendering are covered by contract tests.
- Stage 5 mixes low-page, structure-patch and full-page views. Target patches
  retain full source context and never rebase map coordinates.
- Stage 0 hashes every manifest payload, rejects document leakage, records
  oracle error/provenance/seeds/baseline identity and creates a frozen contract.
  Every formal training start revalidates that contract.
- Historical labels are now traced to a fixed torchvision RAFT-Large generator
  and checkpoint. The migration command permanently classifies them as
  `raft_pseudo`; an independent analytic renderer creates exact Gate-eligible
  maps and document-level splits from flat pages. A Slurm-array renderer and
  fail-closed shard merger support full-corpus preparation without splitting a
  document's variants across workers.
- Stage-specific `deterministic / warr / coordinate_fm / qwen_condition /
  full_page` Python entry points and Bash/Slurm jobs enforce parents and receipts.
- Formal config path binding is environment-explicit and fail-closed:
  Stage 0 manifests/contracts, full-page manifests/contracts, reviewed receipts
  and overridden parent checkpoints use the same `DOCGRID_*` paths in Slurm
  preflight and training, while each resolved value is frozen in the run record.
- Slurm Stage 4/5 wrappers request `DOCGRID_QWEN_HOST_MEM` (192G by default)
  for the frozen 20B Qwen CPU-offload path; deterministic stages retain a 64G
  default and any submitted `--mem` may explicitly override either value.
- A separate `container_jobs/` suite matches the production platform's outer
  allocation plus inner `srun --container-image` model. It runs canonical
  workers directly, avoiding nested `sbatch`, and covers array rendering,
  merge/audit, Qwen validation, Stage 1-5, Gate review receipts, and OCR.
- The same launchers are dual-mode: direct host submission performs one `srun`,
  while a portal wrapper that already entered the Pyxis container is detected
  through the allocation environment and executes the worker locally. This
  fixes the observed job-79121 nested-`srun` exit before any shard was written.
- Stage 5 has an independent, immutable 1024x768 render/merge/audit chain. It
  rejects the known 512x512 source CSV and audits native source resolution and
  aspect before rendering, so an upscaled/stretched corpus cannot masquerade as
  full-page evidence. Gate 5 is bound to this full-page validation manifest.
- Evaluator v3 binds the exact manifest payload and input/output canvas and
  emits EPE/P95/line/edge/straightness/fold/confidence/WARR and
  Jacobian/bending/orthogonality/border metrics, per-sample/group tables,
  L1/PSNR/SSIM/character-edge/table-connectivity metrics, runtime breakdown
  and fixed visual artifacts. Training also uses a valid-mask-aware local SSIM
  auxiliary while map EPE remains primary.
- Gate 1-5 rules are fail-closed; missing baseline, OCR, multi-seed,
  repeatability, efficiency or visual-review evidence cannot produce a passing
  receipt. Gate runs reject runtime ablation overrides.
- Multi-seed aggregation, exact repeatability and named runtime ablation tools
  are implemented. Checkpoints and run manifests record the actual training seed.
- OCR CER/WER scoring verifies the exact geometry sample set and binds its
  report to the checkpoint/manifest/payload hashes. Gate-5 evaluation exports
  every model/oracle/target PNG plus a SHA-bound image manifest; scoring
  re-hashes the files and requires each transcript row to carry the exact
  model/oracle image hashes. Gate 5 rejects changed images, unknown OCR engine
  versions, manually copied numbers and cross-run OCR values.
- The exact frozen 259999 supervised prior now has a decoder-free compatibility
  adapter and same-manifest evaluator-v3 job for Gate 1. Gate 5 similarly fixes
  its deterministic baseline to Stage-2 WARR on the same full-page manifest and
  1024x768 work canvas.

## Local verification evidence

- `95 passed` on Python 3.8 / PyTorch in the RAFT_flow environment, including
  real backward-gradient isolation for Stage 3 and the Stage-4 adapter/fusion.
- A Stage-5 mixed-view CPU epoch completed and recorded all three view types,
  expanded config, run manifest, per-loss metrics and `training_seed=19`.
- Native inference and exact two-run repeatability completed successfully.
- A 15-variant ablation smoke completed and produced immutable per-variant
  evaluations plus `summary.json` and `summary.csv`.
- A 40-page sample of real flat document images was analytically warped into a
  20/13/7 document-disjoint split. Stage 0 returned
  `verified_gt_only=true`, all split fold rates were zero, and the renderer
  oracle mean RGB L1 was `0.0163086`.
- The production sharding path was also executed locally on 40 real CSV rows at
  128x128: two immutable shards merged to 20/13/7 documents, Stage 0 returned
  verified GT/document disjoint/zero folds, and the merged renderer oracle RGB
  L1 was `0.0176721`. The frozen contract SHA-256 is
  `b48de12a7002392f48394fda4f2f6fdb12e2ffc1bf16989df6c83073633cc7cc`.
- A one-epoch Stage-1 probe on that audited dataset completed 10 optimizer
  steps, saved/restored the full checkpoint, and evaluated at
  `EPE=0.596250`, `P95=1.165082`, `fold_rate=0` at 64×64.
- All Slurm Bash files pass `bash -n`.

The analytic probe is verified geometry, but its small sample/resolution and
synthetic warp domain make it engineering evidence only. It cannot pass the
formal full-resolution Gate sequence or establish real-camera quality.

## Still required before the research goal is complete

1. Render and Stage-0-freeze the full intended analytic corpus; add a separately
   verified renderer-GT/real-camera domain set and freeze prior/baseline metrics.
2. Use an environment containing the compatible Qwen-Image-Edit diffusers
   pipeline and validate captured hidden/QK token shapes on the local 20B model.
3. Train Stage 1-5 on the frozen real dataset for seeds 1337, 2027 and 3407,
   respecting every Go/No-Go receipt.
4. Run a fixed OCR engine to create model/oracle transcripts, score them with
   the bound OCR evaluator, and supply hard-subset, efficiency, repeatability
   and visual-review evidence before meeting all Gate-5 thresholds.
5. Complete the formal A1-A20 ablation matrix; the existing 15-variant run is
   only an engineering/runtime smoke suite.
6. Inspect full-page outputs for rewritten/blurred characters, bent table lines,
   periodic ripples, folds, margin loss and crop/scale failures.

Current status is therefore: architecture/training/evaluation infrastructure is
implemented and locally regression-tested; real-data training and Gate-5 quality
acceptance have not yet been completed.
