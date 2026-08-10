#!/usr/bin/env bash
# Single-GPU provenance audit for the two historical typical baselines.
# Submit this foreground script to Slurm; it intentionally stays attached to
# the allocation until both 512/518 historical recipes have been evaluated.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

PYTHON="${PYTHON:-/usr/bin/python}"
TEACHER="${TEACHER:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/exps/26/dewarp/0709_v3v42v2_OriFtGrad10_AugFP32_bigrot/checkpoints/259999_raft_unwarp.pt}"
LAMA="${LAMA:-/juicefs-algorithm/data/IPT/yuang_feng/DATA/warp_test/common_erase.pt}"
SOURCE_DIR="${SOURCE_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/test_silver_bullet_imgs/typical}"
FIRST_DIR="${FIRST_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/test_silver_bullet_imgs/typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999}"
SECOND_DIR="${SECOND_DIR:-/juicefs-algorithm/lts_data/IPT/pengcheng_yu/Dataset/tmp/test_silver_bullet_imgs/typical_0709_v3v42v2_OriFtGrad10_AugFP32_bigrot_259999-2nd}"
# This pair has the largest decoded first-vs-second MAE in the 40-image set,
# making the 512/518 recipe identification less ambiguous than a near-tie.
SAMPLE="${SAMPLE:-16ELhPMMHYV40g57W0XeMfH7_a.jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/teacher_input_size_audit/${SAMPLE%.jpg}}"
THREADS="${THREADS:-32}"

[[ -x "$PYTHON" ]] || { echo "[error] Python is not executable: $PYTHON" >&2; exit 70; }
[[ -s "$TEACHER" ]] || { echo "[error] teacher is missing: $TEACHER" >&2; exit 66; }
[[ -s "$LAMA" ]] || { echo "[error] LAMA is missing: $LAMA" >&2; exit 66; }
[[ -s "$SOURCE_DIR/$SAMPLE" ]] || { echo "[error] source sample is missing: $SOURCE_DIR/$SAMPLE" >&2; exit 66; }
[[ -s "$FIRST_DIR/$SAMPLE" ]] || { echo "[error] first baseline is missing: $FIRST_DIR/$SAMPLE" >&2; exit 66; }
[[ -s "$SECOND_DIR/$SAMPLE" ]] || { echo "[error] second baseline is missing: $SECOND_DIR/$SAMPLE" >&2; exit 66; }
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] THREADS must be a positive integer" >&2; exit 64; }

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

exec "$PYTHON" scripts/audit_teacher_input_size.py \
    --teacher "$TEACHER" \
    --lama "$LAMA" \
    --image "$SOURCE_DIR/$SAMPLE" \
    --reference "$FIRST_DIR/$SAMPLE" \
    --reference "$SECOND_DIR/$SAMPLE" \
    --sizes 512 518 \
    --output-dir "$OUTPUT_DIR" \
    --threads "$THREADS" \
    --device cuda:0
