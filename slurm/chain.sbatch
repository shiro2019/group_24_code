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

# Run step N of a chain file, then queue step N+1 if this step succeeded.
# The QoS allows only 2 submitted jobs per user, so a long pipeline cannot be queued
# up front; each step submits its successor when it finishes instead. While a step
# runs the chain holds one job slot; queueing the next step briefly needs the second
# one, so the submission is retried for a while if that slot is busy.
#   sbatch --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 1
# Each non-empty, non-comment line of the chain file is "<module> <args...>".
# If a step fails or hits the time limit, fix the cause and resubmit from that step,
# e.g. "sbatch --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 3";
# generation resumes where it stopped.

CHAIN="$1"
STEP="${2:-1}"
if ! [[ "$STEP" =~ ^[0-9]+$ ]]; then
    echo "step must be a positive integer, got '$STEP'" >&2
    exit 1
fi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate ca6127
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "$SLURM_SUBMIT_DIR"

mapfile -t STEPS < <(grep -vE '^\s*(#|$)' "$CHAIN" | tr -d '\r')
TOTAL=${#STEPS[@]}
if [ "$STEP" -lt 1 ] || [ "$STEP" -gt "$TOTAL" ]; then
    echo "step $STEP is outside 1..$TOTAL of $CHAIN" >&2
    exit 1
fi
read -r -a CMD <<< "${STEPS[$((STEP - 1))]}"
echo "chain $CHAIN step $STEP/$TOTAL: python -m ${CMD[*]}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m "${CMD[@]}"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "step $STEP failed with exit code $STATUS; chain stopped" >&2
    exit "$STATUS"
fi

if [ "$STEP" -lt "$TOTAL" ]; then
    NEXT=""
    for attempt in 1 2 3 4 5 6; do
        NEXT=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" slurm/chain.sbatch "$CHAIN" $((STEP + 1)))
        [ -n "$NEXT" ] && break
        echo "queueing step $((STEP + 1)) failed (attempt $attempt), retrying in 2 minutes" >&2
        sleep 120
    done
    if [ -z "$NEXT" ]; then
        echo "could not queue step $((STEP + 1)); resubmit it manually" >&2
        exit 1
    fi
    echo "queued step $((STEP + 1)) as job $NEXT"
else
    echo "chain $CHAIN finished"
fi
