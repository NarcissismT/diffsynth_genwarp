#!/usr/bin/env bash
#SBATCH --job-name=mmdit-corr-exp1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Frozen Qwen-Image-Edit correspondence runner for zero-shot and LoRA ablations.
# The selected Slurm node must provide exactly 8 visible datacenter GPUs with
# sufficient BF16 compute capability and memory (A800/H800/A100-80G/H100/etc.).
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
CONFIG="${MMDIT_CONFIG:-$PROJECT_ROOT/configs/mmdit_correspondence_probe.yaml}"
EXPERIMENT_NAME="${MMDIT_EXPERIMENT_NAME:-mmdit_correspondence_exp1_2511_zero_shot}"
EXPERIMENT_MODE="${MMDIT_EXPERIMENT_MODE:-formal_zero_shot}"
RUN_DIR="$(readlink -m "${MMDIT_RUN_DIR:-$PROJECT_ROOT/runs/$EXPERIMENT_NAME}")"
PROFILE="${MMDIT_PROFILE:-full}"
MANIFEST_ROLE="${MMDIT_MANIFEST_ROLE:-validation}"
MODEL_ID="${MMDIT_MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
MODEL_REVISION="${MMDIT_MODEL_REVISION:-}"
PIPELINE_CLASS="${MMDIT_PIPELINE_CLASS:-QwenImageEditPlusPipeline}"
LORA_CHECKPOINT="${MMDIT_LORA_CHECKPOINT:-}"
LORA_ALPHA="${MMDIT_LORA_ALPHA:-32}"
REFERENCE_RUN="${MMDIT_REFERENCE_RUN:-}"
EXPECTED_GPUS=8
MIN_GPU_MEMORY_GIB="${MMDIT_MIN_GPU_MEMORY_GIB:-75}"
MIN_GPU_CC_MAJOR="${MMDIT_MIN_GPU_CC_MAJOR:-8}"

fail() {
    echo "[error] MMDiT correspondence exp1: $*" >&2
    exit 64
}

if [[ "${1:-}" == "--inside" ]]; then
    export MMDIT_CONTAINER_ACTIVE=1
    shift
fi
[[ $# -eq 0 ]] || fail "unknown arguments: $*"

: "${MMDIT_MANIFEST:?Set MMDIT_MANIFEST to a validation/test JSONL manifest}"
[[ "$MANIFEST_ROLE" == "validation" || "$MANIFEST_ROLE" == "test" ]] \
    || fail "MMDIT_MANIFEST_ROLE must be validation or test"
[[ "$PROFILE" == "full" || "$PROFILE" == "pilot" ]] \
    || fail "MMDIT_PROFILE must be full or pilot"
[[ "$MIN_GPU_MEMORY_GIB" =~ ^[0-9]+$ ]] \
    || fail "MMDIT_MIN_GPU_MEMORY_GIB must be a non-negative integer"
[[ "$MIN_GPU_CC_MAJOR" =~ ^[0-9]+$ ]] \
    || fail "MMDIT_MIN_GPU_CC_MAJOR must be a non-negative integer"
[[ -f "$MMDIT_MANIFEST" ]] || fail "manifest not found: $MMDIT_MANIFEST"
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"
MMDIT_MANIFEST="$(readlink -f "$MMDIT_MANIFEST")"
CONFIG="$(readlink -f "$CONFIG")"
export MMDIT_MANIFEST
[[ "$PIPELINE_CLASS" == "QwenImageEditPipeline" || "$PIPELINE_CLASS" == "QwenImageEditPlusPipeline" ]] \
    || fail "unsupported MMDIT_PIPELINE_CLASS: $PIPELINE_CLASS"
[[ "$EXPERIMENT_MODE" == "formal_zero_shot" \
    || "$EXPERIMENT_MODE" == "legacy_base_zero_shot" \
    || "$EXPERIMENT_MODE" == "lora_ablation" ]] \
    || fail "MMDIT_EXPERIMENT_MODE must be formal_zero_shot, legacy_base_zero_shot, or lora_ablation"
if [[ "$EXPERIMENT_MODE" == "formal_zero_shot" ]]; then
    [[ "$PIPELINE_CLASS" == "QwenImageEditPlusPipeline" ]] \
        || fail "formal Exp1 requires QwenImageEditPlusPipeline"
    [[ -z "$LORA_CHECKPOINT" ]] \
        || fail "formal Exp1 forbids LoRA; use run_exp1b_lora_8xA800.sh"
elif [[ "$EXPERIMENT_MODE" == "legacy_base_zero_shot" ]]; then
    [[ "$PIPELINE_CLASS" == "QwenImageEditPipeline" ]] \
        || fail "legacy base zero-shot requires QwenImageEditPipeline"
    [[ -z "$LORA_CHECKPOINT" ]] \
        || fail "legacy base zero-shot forbids LoRA; use run_exp1b_lora_8xA800.sh"
elif [[ "$EXPERIMENT_MODE" == "lora_ablation" ]]; then
    [[ "$PIPELINE_CLASS" == "QwenImageEditPipeline" ]] \
        || fail "the reference LoRA ablation requires QwenImageEditPipeline"
    [[ -n "$LORA_CHECKPOINT" ]] \
        || fail "lora_ablation requires MMDIT_LORA_CHECKPOINT"
fi
[[ -z "$LORA_CHECKPOINT" || -f "$LORA_CHECKPOINT" ]] \
    || fail "LoRA checkpoint not found: $LORA_CHECKPOINT"

if [[ "$MODEL_ID" == /* ]]; then
    [[ -f "$MODEL_ID/model_index.json" ]] \
        || fail "local model is not a complete Diffusers pipeline: $MODEL_ID"
    MODEL_ID="$(readlink -f "$MODEL_ID")"
    local_pipeline_class="$(
        /usr/bin/python3 -c \
            'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "model_index.json").read_text(encoding="utf-8"))["_class_name"])' \
            "$MODEL_ID" 2>/dev/null || true
    )"
    [[ -n "$local_pipeline_class" ]] \
        || fail "cannot read _class_name from $MODEL_ID/model_index.json"
    [[ "$local_pipeline_class" == "$PIPELINE_CLASS" ]] \
        || fail "local model declares $local_pipeline_class, but MMDIT_PIPELINE_CLASS=$PIPELINE_CLASS"
elif [[ "$EXPERIMENT_MODE" == "formal_zero_shot" ]]; then
    [[ "$MODEL_REVISION" =~ ^[0-9a-fA-F]{40}$ ]] \
        || fail "formal Hub-model Exp1 requires a 40-hex immutable MMDIT_MODEL_REVISION; alternatively pass a local snapshot path"
fi
if [[ -n "$LORA_CHECKPOINT" ]]; then
    LORA_CHECKPOINT="$(readlink -f "$LORA_CHECKPOINT")"
fi

export MMDIT_EXPERIMENT_NAME="$EXPERIMENT_NAME"
export MMDIT_EXPERIMENT_MODE="$EXPERIMENT_MODE"
export MMDIT_MODEL_ID="$MODEL_ID"
export MMDIT_MODEL_REVISION="$MODEL_REVISION"
export MMDIT_PIPELINE_CLASS="$PIPELINE_CLASS"
export MMDIT_LORA_CHECKPOINT="$LORA_CHECKPOINT"
export MMDIT_LORA_ALPHA="$LORA_ALPHA"

export HF_HOME="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"
job_key="${SLURM_JOB_ID:-manual}_mmdit_corr"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$job_key/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/$job_key/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$job_key/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/$job_key/xdg_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python_has_runtime() {
    local candidate="$1"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    "$candidate" -c \
        'import os, torch, pyarrow, matplotlib, diffusers; getattr(diffusers, os.environ["MMDIT_PIPELINE_CLASS"]); import peft; from safetensors.torch import load_file' \
        >/dev/null 2>&1
}

python_has_base_runtime() {
    local candidate="$1"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    "$candidate" -c \
        'import torch, pyarrow, matplotlib, diffusers, peft; from safetensors.torch import load_file' \
        >/dev/null 2>&1
}

select_python_runtime() {
    local candidates=()
    local candidate
    if [[ -n "${MMDIT_PYTHON:-}" ]]; then
        candidates+=("$MMDIT_PYTHON")
    else
        candidate="$(command -v python 2>/dev/null || true)"
        [[ -n "$candidate" ]] && candidates+=("$candidate")
        candidate="$(command -v python3 2>/dev/null || true)"
        [[ -n "$candidate" ]] && candidates+=("$candidate")
        candidates+=(/opt/conda/bin/python /usr/local/bin/python /usr/bin/python3 /usr/bin/python)
    fi
    for candidate in "${candidates[@]}"; do
        if python_has_runtime "$candidate"; then
            readlink -f "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$(select_python_runtime || true)"

bootstrap_diffusers_runtime() {
    [[ "$PIPELINE_CLASS" == "QwenImageEditPlusPipeline" ]] || return 1
    [[ "${MMDIT_AUTO_BOOTSTRAP_DIFFUSERS:-1}" == "1" ]] || return 1
    local candidate=""
    local value
    for value in \
        "$(command -v python 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)" \
        /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3 /usr/bin/python; do
        if python_has_base_runtime "$value"; then
            candidate="$(readlink -f "$value")"
            break
        fi
    done
    [[ -n "$candidate" ]] || return 1
    local runtime_dir
    runtime_dir="$(readlink -m "${MMDIT_RUNTIME_DIR:-$HF_HOME/mmdit_correspondence_runtime/diffusers-0.39.0}")"
    mkdir -p "$runtime_dir"
    exec {runtime_lock_fd}>"$runtime_dir.install.lock"
    flock "$runtime_lock_fd"
    if ! PYTHONPATH="$runtime_dir${PYTHONPATH:+:$PYTHONPATH}" \
        "$candidate" -c \
            'import diffusers; assert diffusers.__version__ == "0.39.0"; from diffusers import QwenImageEditPlusPipeline' \
            >/dev/null 2>&1; then
        echo "[info] bootstrapping pinned Diffusers 0.39.0 into $runtime_dir"
        "$candidate" -m pip install \
            --disable-pip-version-check \
            --upgrade \
            --target "$runtime_dir" \
            --no-deps \
            'diffusers==0.39.0'
    fi
    export PYTHONPATH="$runtime_dir${PYTHONPATH:+:$PYTHONPATH}"
    flock -u "$runtime_lock_fd"
    exec {runtime_lock_fd}>&-
    if python_has_runtime "$candidate"; then
        PYTHON="$candidate"
        return 0
    fi
    return 1
}

# The repository's established Slurm image is used when this host-side script
# is not already running inside a platform-selected container.
if [[ -z "$PYTHON" ]]; then
    if [[ "${MMDIT_CONTAINER_ACTIVE:-0}" != "1" ]] && command -v srun >/dev/null 2>&1; then
        container_image="${MMDIT_CONTAINER_IMAGE:-docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
        export MMDIT_CONFIG="$CONFIG"
        export MMDIT_RUN_DIR="$RUN_DIR"
        export MMDIT_PROFILE="$PROFILE"
        export MMDIT_MANIFEST_ROLE="$MANIFEST_ROLE"
        env_names=(
            MMDIT_CONFIG MMDIT_RUN_DIR MMDIT_PROFILE MMDIT_MANIFEST
            MMDIT_MANIFEST_ROLE MMDIT_EXPERIMENT_NAME MMDIT_EXPERIMENT_MODE
            MMDIT_MODEL_ID MMDIT_STAGES MMDIT_MODEL_REVISION MMDIT_PIPELINE_CLASS
            MMDIT_LORA_CHECKPOINT MMDIT_LORA_ALPHA
            MMDIT_REFERENCE_RUN
            MMDIT_LOAD_STAGGER_SECONDS MMDIT_RUN_TESTS
            MMDIT_MIN_GPU_MEMORY_GIB MMDIT_MIN_GPU_CC_MAJOR
            MMDIT_PYTHON MMDIT_CONTAINER_ACTIVE MMDIT_RUNTIME_DIR
            MMDIT_AUTO_BOOTSTRAP_DIFFUSERS PIP_INDEX_URL PIP_TRUSTED_HOST
            HF_HOME TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR
            MPLCONFIGDIR XDG_CACHE_HOME
            TOKENIZERS_PARALLELISM PYTHONUNBUFFERED OMP_NUM_THREADS
        )
        available_names=()
        for name in "${env_names[@]}"; do
            [[ -v "$name" ]] && available_names+=("$name")
        done
        env_csv="$(IFS=,; echo "${available_names[*]}")"
        echo "[info] entering container: $container_image"
        exec srun -K --cpus-per-task=64 \
            --container-image="$container_image" \
            --container-mounts=/juicefs-algorithm:/juicefs-algorithm \
            --container-workdir="$PROJECT_ROOT" \
            --container-env="$env_csv" \
            bash "$SCRIPT_PATH" --inside
    elif [[ "${MMDIT_CONTAINER_ACTIVE:-0}" == "1" ]]; then
        bootstrap_diffusers_runtime || true
    fi
fi
if [[ -z "$PYTHON" ]]; then
    fail "Python lacks torch/diffusers.$PIPELINE_CLASS/peft/safetensors/pyarrow/matplotlib; use the current Diffusers Slurm image"
fi

cd "$PROJECT_ROOT"
mkdir -p \
    "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
    "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_DIR"

if [[ "${MMDIT_RUN_TESTS:-1}" == "1" ]]; then
    if "$PYTHON" -c 'import pytest' >/dev/null 2>&1; then
        "$PYTHON" -m pytest -q \
            tests/test_mmdit_correspondence.py \
            tests/test_mmdit_correspondence_report.py
    else
        "$PYTHON" tools/run_mmdit_correspondence_tests.py
    fi
fi

cuda_probe="$($PYTHON -c 'import torch; count=torch.cuda.device_count(); print(int(torch.cuda.is_available()), count); print("|".join(torch.cuda.get_device_name(i) for i in range(count))); print(" ".join(str(torch.cuda.get_device_properties(i).total_memory // 2**30) for i in range(count))); print(" ".join(f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}" for i in range(count)))')"
cuda_state="$(head -n 1 <<<"$cuda_probe")"
read -r cuda_available cuda_count <<<"$cuda_state"
[[ "$cuda_available" == "1" && "$cuda_count" == "$EXPECTED_GPUS" ]] \
    || fail "expected exactly 8 visible CUDA GPUs, got: $cuda_probe"
gpu_names="$(sed -n '2p' <<<"$cuda_probe")"
gpu_memory="$(sed -n '3p' <<<"$cuda_probe")"
gpu_capabilities="$(sed -n '4p' <<<"$cuda_probe")"
IFS='|' read -r -a gpu_name_values <<<"$gpu_names"
read -r -a gpu_memory_values <<<"$gpu_memory"
read -r -a gpu_capability_values <<<"$gpu_capabilities"
[[ "${#gpu_name_values[@]}" -eq "$EXPECTED_GPUS" \
    && "${#gpu_memory_values[@]}" -eq "$EXPECTED_GPUS" \
    && "${#gpu_capability_values[@]}" -eq "$EXPECTED_GPUS" ]] \
    || fail "incomplete CUDA device properties: $cuda_probe"
for ((gpu_index = 0; gpu_index < EXPECTED_GPUS; gpu_index++)); do
    gpu_name="${gpu_name_values[$gpu_index]}"
    gib="${gpu_memory_values[$gpu_index]}"
    capability="${gpu_capability_values[$gpu_index]}"
    capability_major="${capability%%.*}"
    [[ "$gib" =~ ^[0-9]+$ && "$capability_major" =~ ^[0-9]+$ ]] \
        || fail "invalid CUDA properties for GPU $gpu_index: name=$gpu_name memory=$gib capability=$capability"
    (( gib >= MIN_GPU_MEMORY_GIB )) \
        || fail "GPU $gpu_index ($gpu_name) has ${gib} GiB; require at least ${MIN_GPU_MEMORY_GIB} GiB"
    (( capability_major >= MIN_GPU_CC_MAJOR )) \
        || fail "GPU $gpu_index ($gpu_name) has compute capability $capability; require at least ${MIN_GPU_CC_MAJOR}.0"
done
gpu_type_summary="${gpu_name_values[0]} x${EXPECTED_GPUS}"
for gpu_name in "${gpu_name_values[@]}"; do
    if [[ "$gpu_name" != "${gpu_name_values[0]}" ]]; then
        gpu_type_summary="$gpu_names"
        break
    fi
done
echo "[info] CUDA devices: $gpu_names (${gpu_memory} GiB; compute capabilities $gpu_capabilities)"

exec {experiment_lock_fd}>"$RUN_DIR/.experiment.lock"
flock -n "$experiment_lock_fd" \
    || fail "another job is already writing this run directory: $RUN_DIR"

FROZEN_CONFIG="$RUN_DIR/frozen_config.yaml"
if [[ ! -f "$FROZEN_CONFIG" ]]; then
    build_command=(
        "$PYTHON" tools/build_mmdit_probe_split.py
        --config "$CONFIG"
        --manifest "$MMDIT_MANIFEST"
        --manifest-role "$MANIFEST_ROLE"
        --profile "$PROFILE"
        --run-dir "$RUN_DIR"
        --model-id "$MODEL_ID"
        --pipeline-class "$PIPELINE_CLASS"
        --experiment-name "$EXPERIMENT_NAME"
        --experiment-mode "$EXPERIMENT_MODE"
        --gpu-type "$gpu_type_summary"
    )
    if [[ -n "$MODEL_REVISION" ]]; then
        build_command+=(--model-revision "$MODEL_REVISION")
    fi
    if [[ -n "$LORA_CHECKPOINT" ]]; then
        build_command+=(
            --lora-checkpoint "$LORA_CHECKPOINT"
            --lora-alpha "$LORA_ALPHA"
        )
    fi
    if [[ -n "$REFERENCE_RUN" ]]; then
        build_command+=(--reference-run "$REFERENCE_RUN")
    fi
    "${build_command[@]}"
else
    "$PYTHON" -c '
import pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
model = config["model"]
experiment = config["experiment"]
actual = {
    "experiment_name": str(experiment.get("name", "")),
    "experiment_mode": str(experiment.get("mode", "")),
    "model_id": str(model.get("model_id", "")),
    "revision": str(model.get("revision") or ""),
    "pipeline_class": str(model.get("pipeline_class", "")),
    "lora_checkpoint": str(model.get("lora_checkpoint") or ""),
    "lora_alpha": None if model.get("lora_alpha") is None else float(model["lora_alpha"]),
    "reference_run": str(experiment.get("reference_run") or ""),
    "gpu_type": str(config.get("resources", {}).get("gpu_type") or ""),
}
expected = {
    "experiment_name": sys.argv[2],
    "experiment_mode": sys.argv[3],
    "model_id": sys.argv[4],
    "revision": sys.argv[5],
    "pipeline_class": sys.argv[6],
    "lora_checkpoint": sys.argv[7],
    "lora_alpha": None if not sys.argv[7] else float(sys.argv[8]),
    "reference_run": sys.argv[9],
    "gpu_type": sys.argv[10],
}
if actual != expected:
    raise SystemExit(f"frozen experiment configuration mismatch; use a new MMDIT_RUN_DIR\nactual={actual}\nexpected={expected}")
' "$FROZEN_CONFIG" "$EXPERIMENT_NAME" "$EXPERIMENT_MODE" "$MODEL_ID" "$MODEL_REVISION" "$PIPELINE_CLASS" "$LORA_CHECKPOINT" "$LORA_ALPHA" "$REFERENCE_RUN" "$gpu_type_summary"
    environment_file="$RUN_DIR/environment.json"
    [[ -f "$environment_file" ]] \
        || fail "frozen run is missing environment.json: $environment_file"
    "$PYTHON" -c '
import hashlib, json, pathlib, sys

sys.path.insert(0, sys.argv[7])
from docgrid_flow.analysis.mmdit_correspondence import (
    load_config,
    manifest_asset_paths,
    read_manifest,
    stat_fingerprint,
)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

environment = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
frozen = load_config(sys.argv[2])
all_samples = []
for split_path in frozen["data"]["splits"].values():
    all_samples.extend(read_manifest(split_path))
current_assets = stat_fingerprint(manifest_asset_paths(all_samples))
frozen_model = str(frozen["model"]["model_id"])
current_model = (
    stat_fingerprint([frozen_model]) if pathlib.Path(frozen_model).is_absolute() else None
)
actual = {
    "source_config": str(pathlib.Path(environment["source_config"]).resolve()),
    "source_config_sha256": environment["source_config_sha256"],
    "source_manifest": str(pathlib.Path(environment["source_manifest"]).resolve()),
    "source_manifest_sha256": environment["source_manifest_sha256"],
    "manifest_role": environment["manifest_role_assertion"],
    "profile": environment["profile"],
    "run_dir": str(pathlib.Path(frozen["output"]["run_dir"]).resolve()),
    "asset_stat_fingerprint": environment.get("asset_stat_fingerprint"),
    "model_stat_fingerprint": environment.get("model_stat_fingerprint"),
}
expected = {
    "source_config": str(pathlib.Path(sys.argv[3]).resolve()),
    "source_config_sha256": sha256(sys.argv[3]),
    "source_manifest": str(pathlib.Path(sys.argv[4]).resolve()),
    "source_manifest_sha256": sha256(sys.argv[4]),
    "manifest_role": sys.argv[5],
    "profile": sys.argv[6],
    "run_dir": str(pathlib.Path(sys.argv[8]).resolve()),
    "asset_stat_fingerprint": current_assets,
    "model_stat_fingerprint": current_model,
}
if actual != expected:
    raise SystemExit(
        "frozen data/config identity mismatch; use a new MMDIT_RUN_DIR"
        f"\nactual={actual}\nexpected={expected}"
    )
' "$environment_file" "$FROZEN_CONFIG" "$CONFIG" "$MMDIT_MANIFEST" \
        "$MANIFEST_ROLE" "$PROFILE" "$PROJECT_ROOT" "$RUN_DIR" \
        || fail "refusing to reuse a run directory with changed data/config identity"
    echo "[info] reusing immutable frozen config: $FROZEN_CONFIG"
fi

if [[ -n "$REFERENCE_RUN" ]]; then
    reference_split_dir="$(readlink -m "$REFERENCE_RUN/splits")"
    for split_name in sanity discovery confirmation; do
        current_split="$RUN_DIR/splits/${split_name}_v1.jsonl"
        reference_split="$reference_split_dir/${split_name}_v1.jsonl"
        [[ -f "$reference_split" ]] \
            || fail "reference split not found: $reference_split"
        cmp -s "$current_split" "$reference_split" \
            || fail "$split_name split differs from comparison reference: $REFERENCE_RUN"
    done
    echo "[info] split identity matches comparison reference: $REFERENCE_RUN"
fi

if [[ -n "$LORA_CHECKPOINT" ]]; then
    frozen_lora_sha="$(
        "$PYTHON" -c \
            'import pathlib, sys, yaml; print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["model"]["lora_checkpoint_sha256"])' \
            "$FROZEN_CONFIG"
    )"
    read -r current_lora_sha _ < <(sha256sum -- "$LORA_CHECKPOINT")
    [[ "$current_lora_sha" == "$frozen_lora_sha" ]] \
        || fail "LoRA SHA-256 changed after config freeze: $LORA_CHECKPOINT"
    echo "[info] base model: $MODEL_ID"
    echo "[info] LoRA: $LORA_CHECKPOINT (alpha=$LORA_ALPHA, sha256=$current_lora_sha)"
fi

stages="${MMDIT_STAGES:-all}"
has_stage() {
    [[ "$stages" == "all" || ",${stages}," == *",$1,"* ]]
}

require_sanity_gate() {
    local gate="$RUN_DIR/sanity/sanity_gate.json"
    [[ -f "$gate" ]] || fail "discovery requires a completed sanity gate: $gate"
    "$PYTHON" -c '
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("pass") is not True:
    raise SystemExit(f"sanity gate did not pass: {value}")
' "$gate" || fail "sanity gate did not pass"
}

require_discovery() {
    require_sanity_gate
    [[ -f "$RUN_DIR/discovery/aggregate_metrics.json" ]] \
        || fail "confirmation requires completed discovery aggregation"
    [[ -f "$RUN_DIR/selected_configs.json" ]] \
        || fail "confirmation requires discovery-selected configs"
}

require_confirmation() {
    require_discovery
    [[ -f "$RUN_DIR/confirmation/aggregate_metrics.json" ]] \
        || fail "seed stability requires completed confirmation aggregation"
    "$PYTHON" -c '
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("seed_selection_source") != "confirmation_v1_top3_without_reselection":
    raise SystemExit("seed configs were not ranked by Confirmation")
' "$RUN_DIR/selected_configs.json" || fail "confirmation-ranked seed configs are missing"
}

run_probe() {
    local stage="$1"
    echo "[info] starting $stage on 8 independent one-GPU workers"
    "$PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="$EXPECTED_GPUS" \
        tools/probe_mmdit_correspondence.py \
        --config "$FROZEN_CONFIG" \
        --stage "$stage" \
        --selected-configs "$RUN_DIR/selected_configs.json"
}

run_report() {
    local stage="$1"
    shift
    "$PYTHON" tools/report_mmdit_correspondence.py \
        --config "$FROZEN_CONFIG" \
        --stage "$stage" \
        --expected-ranks "$EXPECTED_GPUS" \
        "$@"
}

if has_stage sanity; then
    sanity_start="$(date +%s)"
    run_probe sanity
    run_report sanity --require-pass
    sanity_seconds="$(( $(date +%s) - sanity_start ))"
    echo "[info] sanity gate passed; measured wall time: ${sanity_seconds}s"
fi

if has_stage discovery; then
    require_sanity_gate
    run_probe discovery
    run_report discovery
fi

if has_stage confirmation; then
    require_discovery
    run_probe confirmation
    run_report confirmation
fi

if has_stage seed_stability; then
    require_confirmation
    run_probe seed_stability
    run_report seed_stability
fi

if [[ "$stages" == "all" ]] || has_stage final; then
    require_confirmation
    [[ -f "$RUN_DIR/seed_stability/seed_stability.json" ]] \
        || fail "final report requires completed seed stability aggregation"
    run_report final
fi

echo "[done] MMDiT correspondence experiment: $RUN_DIR"
echo "[done] report: $RUN_DIR/MMDiT_correspondence_experiment_1_report.md"
