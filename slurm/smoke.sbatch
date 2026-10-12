#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=g24-smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Acceptance check: each method generates 10 images and gets CLIP Score / ImageReward.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate ca6127
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "$SLURM_SUBMIT_DIR"
set -e

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in sdxl_base lcm_lora sdxl_turbo sdxl_lightning; do
    python -m src.generate --config "configs/smoke/$m.yaml"
done
for m in sdxl_base lcm_lora sdxl_turbo sdxl_lightning; do
    python -m src.evaluate --config "configs/smoke/$m.yaml" --metrics clip,imagereward
done
cat outputs/metrics/metrics.csv
