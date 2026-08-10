#!/usr/bin/env bash
#SBATCH --job-name=docgrid-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -n "${DOCGRID_SEEDS:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IFS=',' read -r -a seed_values <<< "${DOCGRID_SEEDS}"
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#seed_values[@]} )); then
    echo "SLURM_ARRAY_TASK_ID is outside DOCGRID_SEEDS" >&2
    exit 2
  fi
  export DOCGRID_SEED="${seed_values[$SLURM_ARRAY_TASK_ID]}"
fi
docgrid_run_stage "${DOCGRID_STAGE:?submit with DOCGRID_STAGE}"

