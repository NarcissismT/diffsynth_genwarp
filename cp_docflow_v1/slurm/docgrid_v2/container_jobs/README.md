# IntSig srun container submission

These are host-side Bash jobs for the platform pattern:

```text
platform allocation -> srun --container-image=... -> canonical worker .sbatch
```

They do not call `sbatch` inside the container. Upload the complete
`cp_docflow_v1/` tree, then submit the files in this directory through the
platform. Paths default to:

```text
data/artifacts: /juicefs-algorithm/data/IPT/zhuochu_yang/docgrid_v2
runs:           /juicefs-algorithm/data/IPT/zhuochu_yang/docgrid_v2/runs
container:      registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers
```

Override `DOCGRID_DATA_ROOT`, `DOCGRID_RUN_ROOT`, or
`DOCGRID_CONTAINER_IMAGE` in the platform environment before submission if
needed. Every output is immutable: choose a new root for a rerun.

Two portal layouts are supported:

1. If the portal executes the selected Bash file on the allocated host, submit
   a `container_jobs/*.sh` file directly; it performs the one required `srun`.
2. If your submitted wrapper already contains
   `srun --container-image=... bash SCRIPT`, set `SCRIPT` to the desired
   `container_jobs/*.sh` file. The shared launcher detects that it is already
   inside the allocation and executes the canonical worker directly. It never
   attempts a nested `srun` or `sbatch`.

Do not add another `srun` *inside* any file in this directory. Job `79121`
demonstrated the old failure mode (`srun is unavailable`, exit code 2); the
dual-mode launcher now handles that portal layout explicitly.

For the portal layout shown in this project, the corrected array body is:

```bash
#!/bin/bash
export HF_HOME=/juicefs-algorithm/data/IPT/yuang_feng/cache
export TRITON_CACHE_DIR=/tmp/slurm_${SLURM_JOB_ID}/triton
export TORCH_EXTENSIONS_DIR=/tmp/slurm_${SLURM_JOB_ID}/deepspeed_cache

srun --cpus-per-task 192 -K \
  --container-image=docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers \
  --container-mounts=/juicefs-algorithm:/juicefs-algorithm \
  --container-workdir=/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp \
  --container-env=HF_HOME,TRITON_CACHE_DIR,TORCH_EXTENSIONS_DIR \
  bash cp_docflow_v1/slurm/docgrid_v2/container_jobs/00_render_full_array_a800.sh
```

Submit it as array indices `0..63`. With the updated shared launcher, the
selected script sees the existing allocation and goes straight to
`render_analytic_gt_shard.sbatch`. Do not point this wrapper at
`00_render_analytic_gt_array.sh`, because that is a login-node `sbatch`
submission helper, not a container worker.

## Submission order

1. Submit `00_render_full_array_a800.sh` as an array with indices `0..63`.
   Each task needs 1 A800, 8 CPU, 64G RAM. If the portal does not provide
   `SLURM_ARRAY_TASK_ID`, submit 64 copies with `DOCGRID_SHARD_INDEX=0..63`.
2. After all 64 tasks succeed, submit `00_merge_cpu.sh` with 8 CPU/32G.
3. Submit `00_evaluate_prior_baseline_a800.sh` with 1 A800, 8 CPU/64G. It
   evaluates the exact frozen 3.47 GB supervised prior on the newly merged
   validation manifest, converts its backward displacement into the v2
   absolute-map contract, and writes immutable Gate-1 baseline metrics.
4. Submit `00_audit_cpu.sh` with 16 CPU/64G. It now requires and verifies the
   prior checkpoint/config/metrics identities, the exact validation-manifest
   hash, every referenced validation payload SHA, and the 512x512 evaluation
   canvas. Do not start Stage 1 unless its
   report says `verified_gt_only=true`, `document_disjoint_verified=true`, and
   all split fold rates are zero.
5. Submit `00_validate_qwen_a800.sh` with 1 A800, 8 CPU, 192G RAM. This freezes
   the real Qwen feature/decoder-free receipt required by Stage 4/5.
   It is independent of steps 1-4 and may run in parallel at any time before
   Stage 4.
6. Submit `01_train_coarse_a800.sh` separately for seeds 1337, 2027, and 3407
   by setting `DOCGRID_SEED` in each platform job.
7. For each reviewed transition, submit `06_evaluate_gate_a800.sh` with
   `DOCGRID_GATE=gate1`, inspect the metric/visual artifacts, then submit
   `06_write_gate_receipt_cpu.sh` with reviewer, evidence and
   `DOCGRID_GATE_DECISION`. A passing immutable receipt unlocks
   `02_train_warr_a800.sh`. The receipt job selects the canonical baseline
   automatically; use `DOCGRID_BASELINE_EVALUATION` only for an intentional
   relocated artifact.
8. Repeat evaluation/review for Stage 2 -> Gate 2 -> Stage 3 -> Gate 3 ->
   Stage 4 -> Gate 4 -> Stage 5. Do not submit every training job in advance;
   the worker intentionally rejects missing or failed receipts.
9. Prepare the independent Stage-5 full-page corpus before Stage 5. Submit
   `00_render_full_page_array_a800.sh` as an array `0..63`, wait for every task,
   then run `00_merge_full_page_cpu.sh` and `00_audit_full_page_cpu.sh` in that
   order. `DOCGRID_FULL_PAGE_INPUT_CSV` is mandatory and must point to flat
   full-resolution pages. The known `metadata_with_flow.csv` is 512x512 and is
   intentionally rejected for this job: upscaling it would neither restore
   character detail nor preserve the 4:3 page aspect. Each shard verifies a
   minimum source size of 1024x768 and target-compatible aspect ratio before
   rendering. Defaults are 20,000 documents, one variant, and 1024x768;
   set `DOCGRID_ANALYTIC_MAX_DOCUMENTS`/`DOCGRID_ANALYTIC_VARIANTS` consistently
   for every array task if changing that immutable run. This preparation may
   run in parallel with Stages 1-4, but its audit must finish before Stage 5.
   Set `DOCGRID_STAGE5_TRAIN_MANIFEST`, `DOCGRID_STAGE5_VAL_MANIFEST`, and
   `DOCGRID_STAGE5_FROZEN_CONTRACT` only when overriding the default paths.
10. After Stage 2 and the full-page audit both finish, submit
    `05_evaluate_deterministic_baseline_a800.sh`. It runs the deterministic
    Stage-2 WARR checkpoint on the same full-page manifest at 1024x768; Gate 5
    rejects any baseline with a different manifest or work canvas.
11. Gate-5 geometry evaluation automatically exports every model/oracle/target
   image plus `gate5_eval/ocr_images.jsonl`. Run a fixed OCR engine on exactly
   the `model_image` and `oracle_image` paths in that file. Build transcript
   CSV/JSONL rows with
   `sample_id/reference_text/model_text/oracle_text/model_image_sha256/oracle_image_sha256`,
   copying both SHA fields from the image manifest. Then submit
   `07_score_ocr_cpu.sh` with explicit `DOCGRID_OCR_TRANSCRIPTS`,
   `DOCGRID_OCR_ENGINE`, and `DOCGRID_OCR_ENGINE_VERSION`; the three geometry
   paths default to the selected seed's Gate-5 evaluation. The scorer rejects
   missing/unknown versions, changed images, cross-run transcripts and sample
   mismatches.

Canonical Gate comparisons are fixed as follows:

| Gate | Current model | Baseline selected by receipt job |
|---|---|---|
| Gate 1 | Stage 1 coarse | frozen supervised TorchScript prior on 512 val |
| Gate 2 | Stage 2 WARR | Stage 1 Gate-1 evaluation |
| Gate 3 | Stage 3 coordinate FM | Stage 2 Gate-2 evaluation |
| Gate 4 | Stage 4 Qwen condition | Stage 3 Gate-3 evaluation |
| Gate 5 | Stage 5 full page | Stage 2 WARR re-evaluated at 1024x768 |

## Gate evaluation example

Set platform environment variables before submitting
`06_evaluate_gate_a800.sh`:

```bash
export DOCGRID_GATE=gate1
export DOCGRID_SEED=1337
```

After reviewing the immutable output, submit `06_write_gate_receipt_cpu.sh`:

```bash
export DOCGRID_GATE=gate1
export DOCGRID_SEED=1337
export DOCGRID_GATE_REVIEWER="$USER"
export DOCGRID_GATE_REVIEW_NOTE="reviewed folds, borders, text/table lines and failures"
export DOCGRID_GATE_DECISION=passed
export DOCGRID_GATE_EVIDENCE=/path/to/reviewed/gate_evidence.json
```

`passed` is not trusted blindly: the Python gate writer rechecks all hard
thresholds and evidence identities and refuses an incomplete receipt.

Gate 1-4 evaluation uses the 512x512 validation manifest. Gate 5 automatically
uses the independently frozen full-page validation manifest; it never silently
falls back to the Stage-1 validation set. Evaluation report schema v3 binds
both actual input/output work sizes; the receipt writer rejects a same-manifest
baseline evaluated at a different resolution or against different payload
assets.

## Local work already completed

The CPU-only real-CSV sharded probe is stored under
`tmp/analytic_sharded_cpu_probe_0731/`. It rendered two shards, merged 40
document-disjoint samples into a 20/13/7 split, and passed Stage 0 with verified
analytic GT and zero folds. Full 116,016-document rendering and Qwen/training
still require the platform jobs above.
