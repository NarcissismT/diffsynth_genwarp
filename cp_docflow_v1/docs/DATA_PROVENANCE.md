# DocGrid-Flow v2 data provenance and GT preparation

Updated 2026-07-31. This is the canonical answer to “which map may be used as
GT?” Formal Gate training must follow this document and the immutable Stage-0
contract; filenames such as `flow_gt_path` are not evidence of ground truth.

## Historical dataset finding

The legacy `metadata_*` flow was produced by `utils/generate_flow_labels.py`,
which runs torchvision RAFT-Large for 20 updates on the corrected/warped image
pair. It is a learned pseudo displacement, not renderer/analytic geometry.

| Asset | SHA-256 |
|---|---|
| `utils/generate_flow_labels.py` | `d914e0b0104c9951c042fc5a0063d073d7a316275561e0dd3bfd133f51605f7a` |
| `utils/flow_utils.py` | `491f3abfdb98a6cb668545af02a2d8b0f89c061c097dde00c4d023b55599aa35` |
| `raft_large_C_T_SKHT_V2-ff5fadd5.pth` | `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322` |

The old image pairs are usually 512×512 while the stored RAFT displacement is
1024×1024. `cp_docflow.migrate_legacy_raft` converts it to the native absolute
source-pixel backward-map convention, but permanently writes
`label_provenance=raft_pseudo` and `gate_eligible=false`. The command cannot
promote the data to `analytic_gt` or `renderer_gt`.

```bash
export DOCGRID_LEGACY_CSV=/path/to/metadata_with_flow.csv
export DOCGRID_MIGRATED_MANIFEST=/new/path/train_raft_pseudo.jsonl
export DOCGRID_MIGRATED_MAP_DIR=/new/path/maps
bash slurm/docgrid_v2/00_migrate_legacy_raft.sh --partition=cpu
```

This migrated data is useful for exploratory pretraining or a pseudo-label
ablation only. It cannot unlock Gate 1, Gate 3, or Gate 5.

## Exact analytic GT path

`cp_docflow.render_analytic_gt` starts from flat document images and generates:

- a perspective plus smooth-bend target-to-source analytic map;
- a warped source image rendered by numerical inverse mapping;
- the exact absolute backward map, valid mask and H/V/boundary labels;
- a resized rectified target copied into the immutable output dataset;
- a hash-based document split before any warp variants are created.

Its declared provenance is `analytic_gt`. It covers controlled analytic
geometry; it does not prove real-camera-domain generalization.

Input CSV fields are `image` and optionally `category`:

```bash
export DOCGRID_PYTHON=/path/to/training-env/bin/python
export DOCGRID_ANALYTIC_INPUT_CSV=/path/to/flat_documents.csv
export DOCGRID_ANALYTIC_OUTPUT=/new/path/docgrid_v2_analytic
export DOCGRID_ANALYTIC_VARIANTS=3
export DOCGRID_RENDER_HEIGHT=512
export DOCGRID_RENDER_WIDTH=512
bash slurm/docgrid_v2/00_render_analytic_gt.sh --partition=a100
```

The currently audited candidate corpus is
`/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_flow.csv`
(116,016 rows). Its `image` column is the flat/rectified source and
`edit_image` is the observed warped image. Its `flow_gt_path` name is legacy:
the files are RAFT pseudo labels and must not be passed to the analytic
renderer or a Gate. A ready-to-copy set of renderer/audit exports is in
`examples/docgrid_full_analytic.env.example`.

All sampled files from this known CSV are 512x512. They may support the
512x512 Stage-1--4 analytic sequence, but they are not native full-page data.
Upscaling them to 1024x768 would invent no detail and would change the aspect
of characters and table cells. Stage 5 therefore requires a different flat
page CSV. `cp_docflow.validate_rectified_sources` reproduces the renderer's
deterministic selection/sharding and checks every selected source for minimum
native resolution and target-compatible aspect before any full-page shard is
written.

For a full corpus, render document-disjoint shards as a Slurm array. Global
hash ordering and `--max-documents` selection happen before round-robin shard
assignment, so every source document and all of its variants belong to exactly
one shard:

```bash
export DOCGRID_ANALYTIC_INPUT_CSV=/path/to/flat_documents.csv
export DOCGRID_ANALYTIC_SHARDS=16
export DOCGRID_ANALYTIC_SHARD_ROOT=/new/path/docgrid_v2_analytic_shards
bash slurm/docgrid_v2/00_render_analytic_gt_array.sh --partition=a100

# Run only after every array task completed successfully.
export DOCGRID_ANALYTIC_MERGED_OUTPUT=/new/path/docgrid_v2_analytic_merged
bash slurm/docgrid_v2/00_merge_analytic_gt.sh --partition=cpu
```

The merge job refuses incomplete/duplicate shard identities, mixed renderer
contracts, manifest hash changes, missing generated assets, duplicate samples,
cross-split documents, or missing variants. It writes new immutable combined
manifests that continue to reference the shard assets. Use the merged output,
not the individual shard manifests, for Stage 0.

Then freeze the exact payload identities:

```bash
# Single-job render: DATA_ROOT="$DOCGRID_ANALYTIC_OUTPUT"
# Array render:      DATA_ROOT="$DOCGRID_ANALYTIC_MERGED_OUTPUT"
export DOCGRID_ANALYTIC_DATA_ROOT=/path/to/the/complete/render-or-merge-output
export DOCGRID_TRAIN_MANIFEST="$DOCGRID_ANALYTIC_DATA_ROOT/manifests/train.jsonl"
export DOCGRID_VAL_MANIFEST="$DOCGRID_ANALYTIC_DATA_ROOT/manifests/val.jsonl"
export DOCGRID_TEST_MANIFEST="$DOCGRID_ANALYTIC_DATA_ROOT/manifests/test.jsonl"
bash slurm/docgrid_v2/00_audit_data.sh --partition=cpu
```

Every formal training start rehashes all referenced images, maps and masks and
rejects drift relative to `frozen_contract.json`.

For Stage 1–4, export the audited `DOCGRID_TRAIN_MANIFEST`,
`DOCGRID_VAL_MANIFEST` and `DOCGRID_FROZEN_CONTRACT` before submitting the
stage scripts. The configs resolve only their documented `DOCGRID_*` variables,
then freeze the resolved paths in the run manifest. Stage 5 uses separately
audited full-page paths named `DOCGRID_STAGE5_TRAIN_MANIFEST`,
`DOCGRID_STAGE5_VAL_MANIFEST` and `DOCGRID_STAGE5_FROZEN_CONTRACT`.

On the production container platform, the canonical Stage-5 data preparation
is `00_render_full_page_array_a800.sh` -> `00_merge_full_page_cpu.sh` ->
`00_audit_full_page_cpu.sh`. It is an independent immutable corpus and may be
prepared while Stages 1--4 train, but its frozen contract must exist before
Stage 5. Gate 5 is bound to its validation manifest rather than the 512x512
validation manifest.

## Verified integration probe

A bounded probe used 40 flat document images selected from the existing
116,016-row corpus, rendered at 64×64 with seed 9. The document split was
20/13/7 train/validation/test; Stage-0 reported:

```text
document_disjoint_verified = true
verified_gt_only = true
fold_rate = 0 for all splits
mean renderer oracle RGB L1 = 0.0163086271
```

A one-epoch Stage-1 CPU probe completed all 10 optimizer steps, checkpointed
successfully and evaluated at `EPE=0.596250`, `P95=1.165082`, `fold_rate=0`.
These small-resolution values prove the data/audit/training/evaluator wiring,
not the formal quality thresholds or real-camera performance.

The later sharded CPU probe used the same 40-document deterministic selection
at 128x128, divided it over two renderer shards, merged the manifests and ran
Stage 0 on the combined result. It retained a 20/13/7 split, reported zero
folds and `verified_gt_only=true`, and produced merged oracle RGB L1
`0.0176720575`. Evidence is under
`tmp/analytic_sharded_cpu_probe_0731/`; merge report SHA-256 is
`04ba96ebfba9761ce531e2e7e2346d8c6577716cca6b7c9bdca0378f56c50fc4`.

## Formal data still required

Before final acceptance, render/audit the full intended analytic corpus and add
a separately verified renderer-GT or real-camera evaluation set with stable
`document_id`, text/language/table/difficulty tags and authoritative maps. Run
all Gate comparisons on unchanged manifests. Keep pseudo, analytic and real
domains in separate group reports even when one training run uses more than one
domain.
