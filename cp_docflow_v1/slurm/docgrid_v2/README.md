# DocGrid-Flow v2 Slurm training

Run every command from `cp_docflow_v1/`. Formal runs are intentionally not
auto-chained: every transition is a reviewed Go/No-Go decision.

If the cluster allocates resources outside the script and expects an inner
`srun --container-image=...` command, use the ready host-side jobs in
`slurm/docgrid_v2/container_jobs/README.md`. Those jobs execute `.sbatch`
workers directly inside the container and never submit nested `sbatch` jobs.
They also detect portals that already wrapped the selected job in `srun` and
avoid a second `srun`; use the exact corrected wrapper example in that README.

## 0. Create or classify labels

Generate exact analytic GT from a flat-document CSV:

```bash
export DOCGRID_PYTHON=/path/to/qwen-env/bin/python
export DOCGRID_ANALYTIC_INPUT_CSV=/path/to/flat_documents.csv
export DOCGRID_ANALYTIC_OUTPUT=/path/to/new/docgrid_v2_analytic
bash slurm/docgrid_v2/00_render_analytic_gt.sh --partition=a100
```

For the known 116,016-row corpus, copy and edit the ready environment template
first:

```bash
cp examples/docgrid_full_analytic.env.example /safe/writable/docgrid.env
# Edit DOCGRID_PYTHON and every /path/to/writable/... value, then:
source /safe/writable/docgrid.env
```

The template deliberately selects CSV column `image` (flat source), not
`edit_image` (legacy warped sample) or `flow_gt_path` (RAFT pseudo label).

For a full corpus, use an array followed by an immutable CPU merge:

```bash
export DOCGRID_ANALYTIC_SHARDS=16
export DOCGRID_ANALYTIC_SHARD_ROOT=/path/to/new/docgrid_v2_analytic_shards
bash slurm/docgrid_v2/00_render_analytic_gt_array.sh --partition=a100

# Submit only after all array tasks are COMPLETED.
export DOCGRID_ANALYTIC_MERGED_OUTPUT=/path/to/new/docgrid_v2_analytic_merged
bash slurm/docgrid_v2/00_merge_analytic_gt.sh --partition=cpu
```

Do not reuse a non-empty shard or merge output. If an array task fails, remove
or relocate only that explicitly identified partial shard before resubmitting
it. The merger requires every `0..N-1` report, identical renderer/data
contracts, unchanged manifest hashes, complete generated assets, exactly the
declared variants per document and non-empty combined train/val/test splits.
Point Stage 0 at the three merged manifests, never at one shard.
`DOCGRID_ANALYTIC_SHARDS` must not exceed the number of documents selected by
`DOCGRID_ANALYTIC_MAX_DOCUMENTS` (or the full CSV count when it is unset).

For historical RAFT displacements, use the separate migration entry point:

```bash
export DOCGRID_LEGACY_CSV=/path/to/metadata_with_flow.csv
export DOCGRID_MIGRATED_MANIFEST=/new/path/train_raft_pseudo.jsonl
export DOCGRID_MIGRATED_MAP_DIR=/new/path/maps
bash slurm/docgrid_v2/00_migrate_legacy_raft.sh --partition=cpu
```

The migration output is always `raft_pseudo` and cannot be used for a Gate.
See `docs/DATA_PROVENANCE.md` for the audited generator/checkpoint hashes.

## 1. Configure the environment and freeze data

```bash
export DOCGRID_PYTHON=/path/to/qwen-env/bin/python
export DOCGRID_TRAIN_MANIFEST=data/train_docgrid_v2.jsonl
export DOCGRID_VAL_MANIFEST=data/val_docgrid_v2.jsonl
export DOCGRID_TEST_MANIFEST=data/test_docgrid_v2.jsonl
bash slurm/docgrid_v2/00_audit_data.sh --partition=a100
```

Stage 0 runs coordinate/data contract tests, hashes every referenced asset,
checks document-level split leakage and writes an immutable
`runs/docgrid_v2/stage0_audit/frozen_contract.json`. Set all three
`DOCGRID_BASELINE_CHECKPOINT`, `DOCGRID_BASELINE_CONFIG` and
`DOCGRID_BASELINE_METRICS` to freeze the deterministic prior identity too.
For the container workflow these are populated by
`00_evaluate_prior_baseline_a800.sh`; Stage 0 verifies that its v3 exploratory
report uses the exact frozen validation manifest, checkpoint/config hashes and
work canvas rather than merely recording three paths. The evaluator also hashes
all referenced images/maps/masks, so unchanged JSONL with replaced assets is
rejected.
Run a separate audit with `DOCGRID_AUDIT_OUTPUT=.../stage0_full_page_audit` and
the full-page manifests before Stage 5.

For the outer-allocation/container platform, the full-page path is explicitly:

```text
00_render_full_page_array_a800.sh (array 0..63)
-> 00_merge_full_page_cpu.sh
-> 00_audit_full_page_cpu.sh
```

Set `DOCGRID_FULL_PAGE_INPUT_CSV` to genuine flat full-resolution pages. The
known 512x512 `metadata_with_flow.csv` is valid for the Stage-1 analytic corpus
but invalid for this path. Every full-page shard requires native dimensions at
least 1024x768 and a 768/1024-compatible width/height ratio before the canonical
renderer runs. This prevents both fake upscaling and character/table stretching.

## Bind the audited data to Stage 1–4

The canonical configs intentionally accept only explicit `DOCGRID_*` path
overrides. Export the same train/validation manifests used by Stage 0 and the
contract it wrote; no YAML editing is needed:

```bash
export DOCGRID_TRAIN_MANIFEST=/path/to/docgrid_v2_analytic_merged/manifests/train.jsonl
export DOCGRID_VAL_MANIFEST=/path/to/docgrid_v2_analytic_merged/manifests/val.jsonl
export DOCGRID_TEST_MANIFEST=/path/to/docgrid_v2_analytic_merged/manifests/test.jsonl
export DOCGRID_FROZEN_CONTRACT="$PWD/runs/docgrid_v2/stage0_audit/frozen_contract.json"
```

If `DOCGRID_AUDIT_OUTPUT` is non-default, set `DOCGRID_FROZEN_CONTRACT` to
`$DOCGRID_AUDIT_OUTPUT/frozen_contract.json` before running the audit; its
Slurm job will then write the audit there. Stage 1–4 preflight and training use
this same variable and rehash the manifests/payloads. Resolved values are
frozen in each run's `config.yaml` and `run_manifest.json`.

For an intentionally relocated parent or reviewed receipt, use the matching
variables such as `DOCGRID_STAGE2_PARENT_CHECKPOINT`,
`DOCGRID_STAGE3_PARENT_CHECKPOINT`, `DOCGRID_GATE1_RECEIPT`, etc. A generic
`DOCGRID_PARENT_CHECKPOINT` is a one-command override and takes precedence for
the submitted stage. The wrapper verifies precisely the path it then passes to
the trainer.

## 2. Train each reviewed stage

```bash
bash slurm/docgrid_v2/01_train_coarse.sh
bash slurm/docgrid_v2/02_train_warr.sh
bash slurm/docgrid_v2/03_train_coordinate_fm.sh
bash slurm/docgrid_v2/04_train_qwen.sh
bash slurm/docgrid_v2/05_finetune_full_page.sh
```

For the required three seeds, use one Slurm array:

```bash
DOCGRID_SEEDS=1337,2027,3407 \
  bash slurm/docgrid_v2/submit_multiseed.sh stage1_coarse --partition=a100
```

The same command accepts `stage2_warr`, `stage3_coordinate_fm`, `stage4_qwen`
and `stage5_full_page`. Parent paths are resolved per seed. `DOCGRID_SEED` also
selects the correct parent for a single-seed submission. Useful overrides are
`DOCGRID_RESUME`, `DOCGRID_PARENT_CHECKPOINT` and `DOCGRID_OUTPUT_DIR`.

The two Qwen stages receive `--mem=${DOCGRID_QWEN_HOST_MEM:-192G}` from the
submission wrappers because the frozen 20B pipeline uses CPU offload. Other
stages default to `--mem=${DOCGRID_STANDARD_HOST_MEM:-64G}`. Pass a later
`--mem=...` to `submit_stage.sh`/`submit_multiseed.sh` to override either
default only after measuring the target cluster; the Qwen validation job
records peak CUDA memory and component parameter count for that decision.

Before Stage 5, audit its full-page manifests independently and bind that
contract explicitly:

```bash
export DOCGRID_STAGE5_TRAIN_MANIFEST=/path/to/full_page/manifests/train.jsonl
export DOCGRID_STAGE5_VAL_MANIFEST=/path/to/full_page/manifests/val.jsonl
export DOCGRID_STAGE5_FROZEN_CONTRACT="$PWD/runs/docgrid_v2/stage0_full_page_audit/frozen_contract.json"
```

`DOCGRID_STAGE5_PARENT_CHECKPOINT` can relocate only the Stage-4 parent without
changing the earlier parent defaults.

Stage 4/5 preflight checks that the local 54 GB directory is a complete
`QwenImageEditPipeline` and that `DOCGRID_PYTHON` can import the matching
diffusers class. It also requires a real feature-hook validation receipt:

```bash
export DOCGRID_QWEN_PROBE_IMAGE=/path/to/representative_warped_page.png
export DOCGRID_QWEN_VALIDATION_REPORT="$PWD/runs/docgrid_v2/qwen_runtime_validation/report.json"
bash slurm/docgrid_v2/00_validate_qwen.sh --partition=a100
```

The GPU validation loads the local model, checks all selected target/source
feature shapes and finite values, replaces `vae.decode` with a forbidden-call
guard, and freezes the model-index/transformer-config hashes. Stage 4/5 refuses
a missing, stale or differently configured receipt. The RAFT_flow smoke
environment intentionally fails this check because it lacks diffusers.

## 3. Evaluate and write a Gate receipt

```bash
DOCGRID_GATE=gate1 \
DOCGRID_CHECKPOINT=runs/docgrid_v2/stage1_coarse/seed-1337/best.pt \
DOCGRID_EVAL_MANIFEST=data/val_docgrid_v2.jsonl \
DOCGRID_EVAL_OUTPUT=runs/docgrid_v2/stage1_coarse/seed-1337/gate_eval \
  bash slurm/docgrid_v2/06_evaluate_gate.sh --partition=a100
```

The evaluation produces `metrics.json`, `per_sample.csv`, confidence
calibration, runtime and fixed visual directories. After it finishes, inspect
the metrics and visual evidence, prepare the external evidence JSON, and write
the immutable receipt separately:

```bash
PYTHONPATH=src "$DOCGRID_PYTHON" -m cp_docflow.gates \
  --gate gate1 \
  --evaluation runs/docgrid_v2/stage1_coarse/seed-1337/gate_eval/metrics.json \
  --baseline-evaluation /path/to/frozen-baseline/metrics.json \
  --evidence /path/to/gate-evidence.json \
  --output runs/docgrid_v2/gates/gate1.json \
  --reviewer "$USER" \
  --review-note "reviewed scale, folds, text/table lines and failures" \
  --passed
```

Do not rerun into the same evaluation directory: evaluations and receipts
refuse to overwrite evidence. `--passed` is refused when any hard metric or
evidence item is absent/failing. Smoke provenance and exploratory reports are
never gate-eligible. Evaluation schema v3 records the actual input/output work
canvas, and baseline comparison rejects a different canvas even when the
manifest path is the same. Use `examples/gate_evidence.example.json` as the
evidence schema reference.

Aggregate three completed evaluation reports with:

```bash
PYTHONPATH=src "$DOCGRID_PYTHON" -m cp_docflow.aggregate_seeds \
  --evaluations /seed-1337/metrics.json /seed-2027/metrics.json /seed-3407/metrics.json \
  --output runs/docgrid_v2/evidence/stage5_multi_seed.json
```

Use `cp_docflow.verify_repeatability` for exact repeated inference and
`cp_docflow.ablation_suite --variants examples/ablation_runtime_v2.json` for
exploratory ablations. Ablation reports cannot be reused as Gate reports.

This implementation is single-process. More GPUs do not enable DDP; Qwen may
still use its configured device map/offload policy. Logs are under
`slurm_logs/docgrid_v2/`.

## 4. Score OCR for Gate 5

Gate-5 geometry evaluation automatically adds `--export-ocr-images` and writes
`ocr_images.jsonl`, `ocr_model_rectified/`, `ocr_oracle_rectified/`, and
`ocr_target_rectified/`. Run one fixed OCR engine directly on the manifest's
model/oracle image paths. Then provide CSV or JSONL fields
`sample_id/reference_text/model_text/oracle_text/model_image_sha256/oracle_image_sha256`.
Copy both image SHA fields from the same `ocr_images.jsonl` row; do not recompute
or substitute images after OCR:

```bash
export DOCGRID_OCR_TRANSCRIPTS=/path/to/transcripts.jsonl
export DOCGRID_OCR_OUTPUT=/path/to/new/ocr_evidence
export DOCGRID_OCR_ENGINE=PaddleOCR
export DOCGRID_OCR_ENGINE_VERSION=3.0.0
export DOCGRID_GEOMETRY_EVALUATION=/path/to/gate_eval/metrics.json
export DOCGRID_GEOMETRY_PER_SAMPLE=/path/to/gate_eval/per_sample.csv
export DOCGRID_OCR_IMAGE_MANIFEST=/path/to/gate_eval/ocr_images.jsonl
bash slurm/docgrid_v2/07_score_ocr.sh --partition=cpu
```

The scorer refuses placeholder/unknown engine versions. It verifies the image
bytes, transcript image hashes, exported-image manifest, geometry sample IDs,
dataset payload, checkpoint and manifest identities, then writes
`gate_evidence.fragment.json`. Merge that fragment with the reviewed boolean
evidence fields. A passing Gate-5 receipt rechecks all of those identities plus
the sample count and CER/WER values.
