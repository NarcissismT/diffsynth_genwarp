#!/usr/bin/env bash
# One-time CPU/login-host preparation for the offline 8xA800 Exp1 job.
# This downloads the pinned Diffusers runtime and the complete 2511 snapshot to
# shared JuiceFS.  It is resumable and must NOT be launched eight times.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
MODEL_REPO="Qwen/Qwen-Image-Edit-2511"
DIFFUSERS_VERSION="0.39.0"
HTTPX_VERSION="0.28.1"
CONTAINER_IMAGE="${MMDIT_PREP_CONTAINER_IMAGE:-registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
export HF_HOME="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"
RUNTIME_DIR="$HF_HOME/mmdit_correspondence_runtime/diffusers-$DIFFUSERS_VERSION"
LOCK_PATH="$HF_HOME/mmdit_correspondence_runtime/prepare-2511.lock"

fail() {
    echo "[error] MMDiT offline-cache preparation: $*" >&2
    exit 64
}

select_python() {
    local candidate
    for candidate in \
        "${MMDIT_PREP_PYTHON:-}" \
        "$(command -v python 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)" \
        /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3; do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        if "$candidate" -c \
            'import huggingface_hub, pip, torch, transformers, safetensors' \
            >/dev/null 2>&1; then
            readlink -f "$candidate"
            return 0
        fi
    done
    return 1
}

if [[ "${1:-}" != "--inside" ]]; then
    [[ $# -eq 0 ]] || fail "unknown arguments: $*"
    if prep_python="$(select_python)"; then
        export MMDIT_PREP_PYTHON="$prep_python"
        exec bash "$SCRIPT_PATH" --inside
    fi

    command -v docker >/dev/null 2>&1 \
        || fail "need either Python with huggingface_hub+pip or Docker on this networked host"
    [[ "$PROJECT_ROOT" == /juicefs-algorithm/* ]] \
        || fail "Docker fallback requires the repository under /juicefs-algorithm"

    docker_env=(-e HF_HOME -e HOME=/tmp/mmdit_prep_home)
    for name in \
        HF_TOKEN HF_ENDPOINT HF_HUB_ENABLE_HF_TRANSFER \
        PIP_INDEX_URL PIP_TRUSTED_HOST \
        HTTPS_PROXY HTTP_PROXY NO_PROXY https_proxy http_proxy no_proxy; do
        [[ -v "$name" ]] && docker_env+=(-e "$name")
    done
    echo "[info] entering preparation container: $CONTAINER_IMAGE"
    exec docker run --rm \
        --user "$(id -u):$(id -g)" \
        "${docker_env[@]}" \
        -v /juicefs-algorithm:/juicefs-algorithm \
        -w /tmp \
        "$CONTAINER_IMAGE" \
        bash "$SCRIPT_PATH" --inside
fi

shift
[[ $# -eq 0 ]] || fail "unknown arguments: $*"
PYTHON="$(select_python || true)"
[[ -n "$PYTHON" ]] || fail "container Python lacks huggingface_hub or pip"

mkdir -p "$HF_HOME" "$RUNTIME_DIR" "$(dirname "$LOCK_PATH")"
exec {prepare_lock_fd}>"$LOCK_PATH"
flock "$prepare_lock_fd"

if ! PYTHONPATH="$RUNTIME_DIR" "$PYTHON" -c \
    'import diffusers; assert diffusers.__version__ == "0.39.0"' \
    >/dev/null 2>&1; then
    echo "[info] installing pinned Diffusers $DIFFUSERS_VERSION into $RUNTIME_DIR"
    "$PYTHON" -m pip install \
        --disable-pip-version-check \
        --upgrade \
        --target "$RUNTIME_DIR" \
        --no-deps \
        "diffusers==$DIFFUSERS_VERSION"
fi

if ! PYTHONPATH="$RUNTIME_DIR" "$PYTHON" -c \
    'import httpx; assert httpx.__version__ == "0.28.1"' \
    >/dev/null 2>&1; then
    echo "[info] installing pinned httpx $HTTPX_VERSION into $RUNTIME_DIR"
    "$PYTHON" -m pip install \
        --disable-pip-version-check \
        --upgrade \
        --target "$RUNTIME_DIR" \
        "httpx==$HTTPX_VERSION"
fi

PYTHONPATH="$RUNTIME_DIR" "$PYTHON" -c \
    'import diffusers; assert diffusers.__version__ == "0.39.0"; from diffusers import QwenImageEditPlusPipeline'

echo "[info] resolving and downloading $MODEL_REPO into $HF_HOME"
MMDIT_PREP_MODEL_REPO="$MODEL_REPO" \
MMDIT_PREP_RUNTIME_DIR="$RUNTIME_DIR" \
MMDIT_PREP_DIFFUSERS_VERSION="$DIFFUSERS_VERSION" \
"$PYTHON" - "$HF_HOME" <<'PY'
import datetime
import json
import os
import pathlib
import sys

from huggingface_hub import HfApi, snapshot_download

hf_home = pathlib.Path(sys.argv[1]).resolve()
hub_cache = hf_home / "hub"
repo_id = os.environ["MMDIT_PREP_MODEL_REPO"]
revision = HfApi().model_info(repo_id).sha
if not revision or len(revision) != 40:
    raise RuntimeError(f"Hub returned an invalid revision for {repo_id}: {revision!r}")

snapshot = pathlib.Path(
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(hub_cache),
        resume_download=True,
    )
).resolve()
model_index = snapshot / "model_index.json"
if not model_index.is_file():
    raise RuntimeError(f"downloaded snapshot lacks model_index.json: {snapshot}")
pipeline_class = json.loads(model_index.read_text(encoding="utf-8")).get("_class_name")
if pipeline_class != "QwenImageEditPlusPipeline":
    raise RuntimeError(
        f"{snapshot} declares {pipeline_class!r}, expected 'QwenImageEditPlusPipeline'"
    )

broken_links = [str(path) for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
if broken_links:
    raise RuntimeError(f"snapshot has broken cache links, first entries: {broken_links[:5]}")

cache_dir = hub_cache / "models--Qwen--Qwen-Image-Edit-2511"
ref_path = cache_dir / "refs" / "main"
ref_path.parent.mkdir(parents=True, exist_ok=True)
ref_tmp = ref_path.with_suffix(".tmp")
ref_tmp.write_text(revision + "\n", encoding="utf-8")
ref_tmp.replace(ref_path)

receipt = {
    "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repo_id": repo_id,
    "revision": revision,
    "snapshot": str(snapshot),
    "pipeline_class": pipeline_class,
    "diffusers_version": os.environ["MMDIT_PREP_DIFFUSERS_VERSION"],
    "diffusers_runtime": os.environ["MMDIT_PREP_RUNTIME_DIR"],
}
receipt_path = hf_home / "mmdit_correspondence_runtime" / "qwen-image-edit-2511.json"
receipt_tmp = receipt_path.with_suffix(".tmp")
receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt_tmp.replace(receipt_path)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "[ok] offline cache is ready"
echo "[ok] receipt: $HF_HOME/mmdit_correspondence_runtime/qwen-image-edit-2511.json"
echo "[next] submit the unchanged 8xA800 job with:"
echo "[next] bash slurm/mmdit_correspondence/container_jobs/00_run_exp1_8xA800.sh"
