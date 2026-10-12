#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=g24-chain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run step N of a chain file, after first queueing step N+1 behind this job.
# The QoS allows only 2 submitted jobs per user, so a long pipeline cannot be queued
# up front; each step submits its successor when it starts instead.
#   sbatch --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 1
# Each non-empty, non-comment line of the chain file is "<module> <args...>".

CHAIN="$1"
STEP="${2:-1}"

module load anaconda
eval "$(conda shell.bash hook)"
conda activate ca6127
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "$SLURM_SUBMIT_DIR"

mapfile -t STEPS < <(grep -vE '^\s*(#|$)' "$CHAIN")
TOTAL=${#STEPS[@]}
CMD="${STEPS[$((STEP - 1))]}"
echo "chain $CHAIN step $STEP/$TOTAL: python -m $CMD"

if [ "$STEP" -lt "$TOTAL" ]; then
    NEXT=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
        --job-name="$SLURM_JOB_NAME" slurm/chain.sbatch "$CHAIN" $((STEP + 1)))
    echo "queued step $((STEP + 1)) as job $NEXT"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m $CMD
