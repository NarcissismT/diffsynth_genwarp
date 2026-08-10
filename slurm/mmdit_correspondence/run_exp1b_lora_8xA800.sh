#!/usr/bin/env bash
#SBATCH --job-name=mmdit-corr-exp1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Frozen ablation matching scripts/train_exp_A_layer12.sh:
# local Qwen-Image-Edit base + rank-32 step-668000 PEFT LoRA.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
MODEL_ROOT_DEFAULT="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/models/Qwen-Image-Edit"

# This is the same model payload listed explicitly in
# scripts/train_exp_A_layer12.sh.  Diffusers reads model_index.json plus the
# transformer/text-encoder shard index files, so no checkpoint conversion or
# Hub re-download occurs.
model_root="${MMDIT_MODEL_ID:-$MODEL_ROOT_DEFAULT}"
required_model_files=(
    "$model_root/model_index.json"
    "$model_root/transformer/diffusion_pytorch_model.safetensors.index.json"
    "$model_root/text_encoder/model.safetensors.index.json"
    "$model_root/vae/diffusion_pytorch_model.safetensors"
    "$model_root/tokenizer/tokenizer_config.json"
    "$model_root/processor/preprocessor_config.json"
)
for required_file in "${required_model_files[@]}"; do
    [[ -f "$required_file" ]] || {
        echo "[error] reference Qwen-Image-Edit payload is incomplete: $required_file" >&2
        exit 64
    }
done

export MMDIT_EXPERIMENT_NAME="${MMDIT_EXPERIMENT_NAME:-mmdit_correspondence_exp1b_step668000_lora}"
export MMDIT_EXPERIMENT_MODE=lora_ablation
export MMDIT_RUN_DIR="${MMDIT_RUN_DIR:-$SCRIPT_DIR/../../runs/$MMDIT_EXPERIMENT_NAME}"
export MMDIT_MODEL_ID="$model_root"
export MMDIT_PIPELINE_CLASS=QwenImageEditPipeline
export MMDIT_LORA_CHECKPOINT="${MMDIT_LORA_CHECKPOINT:-/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio/result/20250929-1_1in10_w_unwarp/step-668000.safetensors}"
export MMDIT_LORA_ALPHA="${MMDIT_LORA_ALPHA:-32}"

exec bash "$SCRIPT_DIR/run_exp1_8xA800.sh" "$@"
