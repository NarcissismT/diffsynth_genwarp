#!/usr/bin/env bash
# Slurm job body: 1 GPU, 16 CPU, 12 hours recommended.
# Run inside docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

d2r_cd_project
d2r_require_visible_gpus 1
d2r_verify_seed
d2r_acquire_job_lock production_pipeline

echo "[info] 开始 production teacher-capacity audit：seed=$D2R_SEED_CHECKPOINT"
generation_json="$(
    env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
        "$D2R_PYTHON" scripts/teacher_capacity_production.py generate \
        --config "$D2R_CONFIG" \
        --pointer "$D2R_CAPACITY_POINTER" \
        --resume "$D2R_SEED_CHECKPOINT" \
        --output-dir "$D2R_CAPACITY_DIR" \
        --seed 42 \
        --batch-size 1 \
        --device cuda:0 \
        --threads 16
)"

d2r_verify_seed
d2r_require_capacity_pointer
generated_receipt="$(
    "$D2R_PYTHON" -c '
import json, pathlib, re, sys
payload = json.loads(sys.argv[1])
required = {"evidence_path", "pointer_path", "receipt_sha256", "receipt_b64"}
if set(payload) != required:
    raise SystemExit(f"unexpected generate result keys: {sorted(payload)}")
if pathlib.Path(payload["pointer_path"]).resolve() != pathlib.Path(sys.argv[2]).resolve():
    raise SystemExit("generate result points to a different approved.json")
if re.fullmatch(r"[0-9a-f]{64}", payload["receipt_sha256"]) is None:
    raise SystemExit("generate result has invalid receipt_sha256")
if re.fullmatch(r"[A-Za-z0-9_-]+", payload["receipt_b64"]) is None:
    raise SystemExit("generate result has invalid receipt_b64")
print(payload["receipt_b64"])
' "$generation_json" "$D2R_CAPACITY_POINTER"
)" || d2r_fail "generate 成功输出未通过严格解析"
d2r_verify_receipt_text "$generated_receipt"

verified_receipt="$(
    env PYTHONPATH="$D2R_PROJECT_ROOT/src" \
        "$D2R_PYTHON" scripts/teacher_capacity_production.py verify \
        --config "$D2R_CONFIG" \
        --pointer "$D2R_CAPACITY_POINTER" \
        --resume "$D2R_SEED_CHECKPOINT"
)" || d2r_fail "刚发布的 production capacity evidence 无法复验"
d2r_verify_receipt_text "$verified_receipt"
[[ "$verified_receipt" == "$generated_receipt" ]] \
    || d2r_fail "generate 与 verify 返回的 receipt 不一致"

receipt_sha="$(
    "$D2R_PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["receipt_sha256"])' \
        "$generation_json"
)"
echo "D2R_V33_CAPACITY_PASS pointer=$D2R_CAPACITY_POINTER receipt_sha256=$receipt_sha"
