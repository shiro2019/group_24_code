#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=g24
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run any module of this repo on one GPU, from the repo root:
#   sbatch --job-name=e2-base slurm/job.sbatch src.generate --config configs/e2/sdxl_base.yaml
# Pin a GPU type for timing runs with --nodelist, e.g. --nodelist=TC2N07 (L40S).

module load anaconda
eval "$(conda shell.bash hook)"
conda activate ca6127
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m "$@"
