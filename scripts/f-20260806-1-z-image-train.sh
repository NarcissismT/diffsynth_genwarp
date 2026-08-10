#!/usr/bin/env bash
set -euo pipefail

# Verified against DiffSynth-Studio 2.1.0 (main commit 899d2cd, 2026-08-07).
# Z-Image-Omni-Base is used because this is an image-editing dataset.
# Plain Tongyi-MAI/Z-Image is text-to-image and does not consume edit_image.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
diffsynth_root="${DIFFSYNTH_STUDIO_ROOT:-$project_root/third_party/DiffSynth-Studio}"

csv_path="${CSV_PATH:-/juicefs-algorithm/data/IPT/yuang_feng/DATA/upwarp_img_1in10_white/1in10_w_metadata.csv}"
train_id="${TRAIN_ID:-20260806-1_1in10_w_z_image_omni_base_unwarp}"
output_root="${OUTPUT_ROOT:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result}"
accelerate_config="${ACCELERATE_CONFIG:-$script_dir/Acceconfig_8A800.yaml}"
model_dir="${Z_IMAGE_MODEL_DIR:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Z-Image-Omni-Base}"

# Compute nodes must use the pre-downloaded local payload only.
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

train_script="$diffsynth_root/examples/z_image/model_training/train.py"
if [[ ! -f "$train_script" || ! -f "$diffsynth_root/diffsynth/pipelines/z_image.py" ]]; then
  printf '%s\n' \
    "Error: Z-Image training support was not found in: $diffsynth_root" \
    "Use a current DiffSynth-Studio checkout and set:" \
    "  DIFFSYNTH_STUDIO_ROOT=/path/to/DiffSynth-Studio $0" >&2
  exit 1
fi

if [[ ! -r "$csv_path" ]]; then
  printf 'Error: dataset metadata is not readable: %s\n' "$csv_path" >&2
  exit 1
fi

if [[ ! -r "$accelerate_config" ]]; then
  printf 'Error: Accelerate config is not readable: %s\n' "$accelerate_config" >&2
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  printf '%s\n' 'Error: accelerate is not installed or is not on PATH.' >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Error: Python is not installed or is not on PATH: %s\n' "$python_bin" >&2
  exit 1
fi

# Build the model_paths JSON from the local shard indexes and fail before
# launching eight workers if any file is absent or is not a valid safetensors file.
model_paths_json="$("$python_bin" - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])


def indexed_shards(folder, index_name, fallback_pattern):
    index_path = folder / index_name
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        paths = [folder / name for name in names]
    else:
        paths = sorted(folder.glob(fallback_pattern))
    if not paths:
        raise SystemExit(f"Error: no weight shards found in {folder}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Error: missing model shards:\n  " + "\n  ".join(missing))
    return paths


transformer = indexed_shards(
    root / "transformer",
    "diffusion_pytorch_model.safetensors.index.json",
    "*.safetensors",
)
text_encoder = indexed_shards(
    root / "text_encoder",
    "model.safetensors.index.json",
    "*.safetensors",
)
siglip = root / "siglip/model.safetensors"
vae = root / "vae/diffusion_pytorch_model.safetensors"
tokenizer_config = root / "tokenizer/tokenizer_config.json"

required = [siglip, vae, tokenizer_config]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Error: incomplete Z-Image-Omni-Base payload:\n  " + "\n  ".join(missing))

for path in [*transformer, siglip, *text_encoder, vae]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            next(iter(handle.keys()))
    except Exception as exc:
        raise SystemExit(f"Error: invalid safetensors file {path}: {exc}") from exc

print(json.dumps([
    [str(path) for path in transformer],
    str(siglip),
    [str(path) for path in text_encoder],
    str(vae),
]))
PY
)"

mkdir -p "$output_root/$train_id"
cd "$diffsynth_root"

# Make the pinned checkout take precedence over the incomplete legacy copy.
export PYTHONPATH="$diffsynth_root${PYTHONPATH:+:$PYTHONPATH}"

accelerate launch --config_file "$accelerate_config" "$train_script" \
  --dataset_base_path "$(dirname -- "$csv_path")" \
  --dataset_metadata_path "$csv_path" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 5 \
  --tokenizer_path "$model_dir/tokenizer" \
  --model_paths "$model_paths_json" \
  --learning_rate 1e-4 \
  --num_epochs 6 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$output_root/$train_id" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 2 \
  --find_unused_parameters \
  --save_steps 4000
