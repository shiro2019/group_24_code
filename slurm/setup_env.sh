#!/bin/bash
# One-off environment setup on the head node (package installation only, no computation).
#   bash slurm/setup_env.sh
set -e
module load anaconda
eval "$(conda shell.bash hook)"
if conda env list | grep -q "^ca6127 "; then
    conda env update -n ca6127 -f environment.yml --prune
else
    conda env create -f environment.yml
fi
mkdir -p logs
