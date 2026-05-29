#!/bin/bash
#SBATCH --job-name=ea-v19-scaled
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/ea-v19-scaled-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/ea-v19-scaled-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/activation-harvester/src \
    $PY scripts/ea_phase2_v19_scaled.py \
    --magnitude 1.5 \
    --n-rollouts 25 --max-new-tokens 300 --tp-size 1 \
    --mem-fraction-static 0.88 --max-running-requests 32 \
    --n-bins 21 --save-dim 3 \
    --out-dir figures/ea_phase2_v19_scaled
