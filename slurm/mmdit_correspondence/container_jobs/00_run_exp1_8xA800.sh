#!/usr/bin/env bash
# Container-side entrypoint for an allocated 8x datacenter-GPU Slurm job.
# Launch this file through the portal's outer `srun --container-image ...`.
# All experiment settings intentionally live here: the outer command only
# passes HF_HOME, TRITON_CACHE_DIR, and TORCH_EXTENSIONS_DIR.
set -Eeuo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$JOB_DIR/../../.." && pwd -P)"

# ---------------------------------------------------------------------------
# Frozen step-668000 LoRA comparison.  The model payload and adapter match
# scripts/f-20250929-1-train.sh and scripts/train_exp_A_layer12.sh.
# ---------------------------------------------------------------------------
readonly MMDIT_MANIFEST_DEFAULT="$PROJECT_ROOT/artifacts/mmdit_correspondence/analytic_seed1337_512/manifests/validation.jsonl"
readonly MMDIT_RUN_DIR_DEFAULT="/juicefs-algorithm/data/IPT/zhuochu_yang/mmdit_correspondence/runs/mmdit_correspondence_exp1b_step668000_lora"
readonly MMDIT_MODEL_ID_DEFAULT="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"
readonly MMDIT_LORA_CHECKPOINT_DEFAULT="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors"
readonly MMDIT_LORA_SHA256="6604490b166a61c098fb3e428f59afb8de28d47c5a77fee6d92391072ee66e44"
readonly MMDIT_REFERENCE_RUN_DEFAULT="/juicefs-algorithm/data/IPT/zhuochu_yang/mmdit_correspondence/runs/mmdit_correspondence_exp1_legacy_base_zero_shot"

export MMDIT_MANIFEST="$MMDIT_MANIFEST_DEFAULT"
export MMDIT_MANIFEST_ROLE=validation
export MMDIT_PROFILE=full
export MMDIT_RUN_DIR="$MMDIT_RUN_DIR_DEFAULT"
export MMDIT_MODEL_ID="$MMDIT_MODEL_ID_DEFAULT"
export MMDIT_MODEL_REVISION=""
export MMDIT_EXPERIMENT_NAME=mmdit_correspondence_exp1b_step668000_lora
export MMDIT_EXPERIMENT_MODE=lora_ablation
export MMDIT_PIPELINE_CLASS=QwenImageEditPipeline
export MMDIT_LORA_CHECKPOINT="$MMDIT_LORA_CHECKPOINT_DEFAULT"
export MMDIT_LORA_ALPHA=32
export MMDIT_REFERENCE_RUN="$MMDIT_REFERENCE_RUN_DEFAULT"
export MMDIT_STAGES=all
export MMDIT_RUN_TESTS=1
export MMDIT_MIN_GPU_MEMORY_GIB=75
export MMDIT_MIN_GPU_CC_MAJOR=8
export MMDIT_AUTO_BOOTSTRAP_DIFFUSERS=0
export MMDIT_CONTAINER_ACTIVE=1
export HF_HOME="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
job_key="${SLURM_JOB_ID:-manual}_mmdit_corr"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$job_key/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/$job_key/torch_extensions}"

[[ -f "$MMDIT_MANIFEST" ]] || {
    echo "[error] formal MMDiT manifest not found: $MMDIT_MANIFEST" >&2
    echo "[error] first run: bash $PROJECT_ROOT/scripts/prepare_mmdit_manifest_cpu.sh" >&2
    echo "[error] instructions: $PROJECT_ROOT/slurm/mmdit_correspondence/README.md" >&2
    exit 64
}
[[ -f "$MMDIT_LORA_CHECKPOINT" ]] || {
    echo "[error] step-668000 LoRA checkpoint not found: $MMDIT_LORA_CHECKPOINT" >&2
    exit 64
}
for reference_file in \
    "$MMDIT_REFERENCE_RUN/frozen_config.yaml" \
    "$MMDIT_REFERENCE_RUN/splits/sanity_v1.jsonl" \
    "$MMDIT_REFERENCE_RUN/splits/discovery_v1.jsonl" \
    "$MMDIT_REFERENCE_RUN/splits/confirmation_v1.jsonl"; do
    [[ -f "$reference_file" ]] || {
        echo "[error] completed base reference is missing: $reference_file" >&2
        exit 64
    }
done

# Slurm compute nodes remain fully offline.  Validate every local pipeline
# component and all shards before the eight workers each load a model copy.
runtime_python="$(command -v python || command -v python3 || true)"
[[ -n "$runtime_python" ]] || {
    echo "[error] Python is unavailable in the experiment container" >&2
    exit 64
}
required_model_files=(
    "$MMDIT_MODEL_ID/model_index.json"
    "$MMDIT_MODEL_ID/scheduler/scheduler_config.json"
    "$MMDIT_MODEL_ID/transformer/config.json"
    "$MMDIT_MODEL_ID/transformer/diffusion_pytorch_model.safetensors.index.json"
    "$MMDIT_MODEL_ID/text_encoder/config.json"
    "$MMDIT_MODEL_ID/text_encoder/model.safetensors.index.json"
    "$MMDIT_MODEL_ID/vae/config.json"
    "$MMDIT_MODEL_ID/vae/diffusion_pytorch_model.safetensors"
    "$MMDIT_MODEL_ID/tokenizer/tokenizer_config.json"
    "$MMDIT_MODEL_ID/processor/preprocessor_config.json"
)
for required_file in "${required_model_files[@]}"; do
    [[ -f "$required_file" ]] || {
        echo "[error] local Qwen-Image-Edit payload is incomplete: $required_file" >&2
        exit 66
    }
done

cached_pipeline_class="$($runtime_python -c \
    'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "model_index.json").read_text(encoding="utf-8"))["_class_name"])' \
    "$MMDIT_MODEL_ID" 2>/dev/null || true)"
[[ "$cached_pipeline_class" == "$MMDIT_PIPELINE_CLASS" ]] || {
    echo "[error] cached model declares ${cached_pipeline_class:-<unreadable>}, expected $MMDIT_PIPELINE_CLASS" >&2
    echo "[error] model: $MMDIT_MODEL_ID" >&2
    exit 65
}
if ! "$runtime_python" -c \
    'import json, pathlib, sys; missing=[]
for value in sys.argv[1:]:
    index=pathlib.Path(value); data=json.loads(index.read_text(encoding="utf-8"))
    missing.extend(str(index.parent / name) for name in set(data["weight_map"].values()) if not (index.parent / name).is_file())
assert not missing, missing[:10]' \
    "$MMDIT_MODEL_ID/transformer/diffusion_pytorch_model.safetensors.index.json" \
    "$MMDIT_MODEL_ID/text_encoder/model.safetensors.index.json"; then
    echo "[error] one or more local Qwen-Image-Edit weight shards are missing" >&2
    exit 66
fi
if ! "$runtime_python" -c 'from diffusers import QwenImageEditPipeline' >/dev/null 2>&1; then
    echo "[error] container Diffusers cannot import QwenImageEditPipeline" >&2
    exit 70
fi
lora_sha256="$(sha256sum "$MMDIT_LORA_CHECKPOINT" | awk '{print $1}')"
[[ "$lora_sha256" == "$MMDIT_LORA_SHA256" ]] || {
    echo "[error] unexpected step-668000 LoRA SHA-256: $lora_sha256" >&2
    echo "[error] expected: $MMDIT_LORA_SHA256" >&2
    exit 65
}
if ! PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$runtime_python" -c '
import sys
from safetensors.torch import load_file
from docgrid_flow.providers.qwen_diffusers import _normalize_lora_state_dict
state = load_file(sys.argv[1], device="cpu")
normalized, rank, targets = _normalize_lora_state_dict(state)
expected = {
    "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
    "to_out.0", "to_add_out", "img_mlp.net.2", "img_mod.1",
    "txt_mlp.net.2", "txt_mod.1",
}
assert len(normalized) == 1440, len(normalized)
assert rank == 32, rank
assert set(targets) == expected, (targets, sorted(expected))
' "$MMDIT_LORA_CHECKPOINT"; then
    echo "[error] step-668000 checkpoint failed strict LoRA structure validation" >&2
    exit 65
fi

echo "[info] manifest: $MMDIT_MANIFEST"
echo "[info] model: $MMDIT_MODEL_ID (local base, offline)"
echo "[info] LoRA: $MMDIT_LORA_CHECKPOINT (rank=32, alpha=$MMDIT_LORA_ALPHA, sha256=$lora_sha256)"
echo "[info] comparison reference: $MMDIT_REFERENCE_RUN"
echo "[info] output: $MMDIT_RUN_DIR"

exec bash "$PROJECT_ROOT/slurm/mmdit_correspondence/run_exp1_8xA800.sh" --inside
