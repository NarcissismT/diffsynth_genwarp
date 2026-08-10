#!/usr/bin/env bash
# Generate the formal MMDiT analytic validation manifest locally on CPU.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
SOURCE_CSV="/juicefs-algorithm/data/IPT/zhuochu_yang/upwarp_img_1in10_white_0511/metadata_with_flow.csv"
OUTPUT_ROOT="$PROJECT_ROOT/artifacts/mmdit_correspondence/analytic_seed1337_512"
RAW_VAL="$OUTPUT_ROOT/manifests/val.jsonl"
FORMAL_MANIFEST="$OUTPUT_ROOT/manifests/validation.jsonl"
CONTAINER_IMAGE="registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers"

[[ -f "$SOURCE_CSV" ]] || {
    echo "[error] flat-document source CSV not found: $SOURCE_CSV" >&2
    exit 64
}
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "[error] immutable output already exists: $OUTPUT_ROOT" >&2
    echo "[error] inspect it or choose a new OUTPUT_ROOT in this script; it will not be overwritten" >&2
    exit 64
fi

mkdir -p "$(dirname "$OUTPUT_ROOT")"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v /juicefs-algorithm:/juicefs-algorithm \
    -w "$PROJECT_ROOT" \
    -e PYTHONPATH="$PROJECT_ROOT/cp_docflow_v1/src" \
    "$CONTAINER_IMAGE" \
    python -m cp_docflow.render_analytic_gt \
        --input-csv "$SOURCE_CSV" \
        --output-dir "$OUTPUT_ROOT" \
        --image-column image \
        --category-column category \
        --variants-per-document 1 \
        --cycle-profile-by-document \
        --seed 1337 \
        --train-ratio 0.05 \
        --val-ratio 0.90 \
        --test-ratio 0.05 \
        --output-height 512 \
        --output-width 512 \
        --max-documents 450 \
        --device cpu

python "$PROJECT_ROOT/tools/finalize_mmdit_analytic_manifest.py" \
    --input "$RAW_VAL" \
    --output "$FORMAL_MANIFEST" \
    --minimum-documents 328

echo "[done] formal MMDiT validation manifest: $FORMAL_MANIFEST"
