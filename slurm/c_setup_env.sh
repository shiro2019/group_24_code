#!/bin/bash
# Alternative setup: conda-forge Python (no defaults channel) plus the pip pins from environment.yml.
# Run on the head node for package installation only, not model computation.
set -eo pipefail
module load anaconda
eval "$(conda shell.bash hook)"
mkdir -p logs
if ! conda env list | grep -q '^ca6127 '; then
    conda create -y -n ca6127 --override-channels -c conda-forge python=3.10 pip
fi
# Extract the pip list exactly as pinned in environment.yml.
awk '/^  - pip:/{flag=1;next} flag && /^      - / {sub(/^      - /, "");print}' \
    environment.yml > logs/c_requirements.txt
test -s logs/c_requirements.txt
conda activate ca6127
python -m pip install --no-cache-dir -r logs/c_requirements.txt
python -m pip check
