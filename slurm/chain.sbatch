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

# Run steps FROM..TO of a chain file one after another inside this job, stopping at the
# first failure. Compute nodes cannot submit jobs, so a pipeline longer than the 6-hour
# limit is split into at most two jobs submitted from the login node, the second one
# waiting for the first:
#   J1=$(sbatch --parsable --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 1 2)
#   sbatch --dependency=afterok:$J1 --kill-on-invalid-dep=yes --job-name=e2-base \
#       slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 3 4
# Each non-empty line of the chain file that does not start with "#" is
# "<module> <args...>". Arguments are split on whitespace: quotes are not interpreted
# and a trailing "# comment" would be passed as arguments.
# If a step fails or the job hits the time limit, the dependent job is cancelled too.
# Resubmit the same ranges starting from the failed step, keeping an explicit end step
# so each job stays under 6 h; e.g. if step 2 failed:
#   J1=$(sbatch --parsable --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 2 2)
#   sbatch --dependency=afterok:$J1 --kill-on-invalid-dep=yes --job-name=e2-base \
#       slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 3 4
# Generation resumes where it stopped.

CHAIN="$1"
FROM="${2:-1}"

module load anaconda
eval "$(conda shell.bash hook)"
conda activate ca6127
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "$SLURM_SUBMIT_DIR"

mapfile -t STEPS < <(grep -vE '^\s*(#|$)' "$CHAIN" | tr -d '\r')
TOTAL=${#STEPS[@]}
TO="${3:-$TOTAL}"
for n in "$FROM" "$TO"; do
    if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -lt 1 ] || [ "$n" -gt "$TOTAL" ]; then
        echo "steps must be integers within 1..$TOTAL of $CHAIN, got '$n'" >&2
        exit 1
    fi
done
if [ "$FROM" -gt "$TO" ]; then
    echo "FROM ($FROM) is after TO ($TO)" >&2
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for ((step = FROM; step <= TO; step++)); do
    read -r -a CMD <<< "${STEPS[$((step - 1))]}"
    echo "chain $CHAIN step $step/$TOTAL: python -m ${CMD[*]}"
    python -m "${CMD[@]}"
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "step $step failed with exit code $status; stopped" >&2
        exit "$status"
    fi
done
echo "steps $FROM-$TO of $CHAIN finished"
