#!/bin/bash
#SBATCH --job-name=ea-phase2-v8
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/ea-phase2-v8-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/ea-phase2-v8-%j.err
#SBATCH --dependency=afterany:14706

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/activation-harvester/src \
    $PY scripts/ea_phase2_v8_sglang.py \
    --magnitudes 0.5 1.0 1.5 2.0 3.0 5.0 \
    --n-rollouts 8 --max-new-tokens 80 --tp-size 1 \
    --mem-fraction-static 0.88 --max-running-requests 16 \
    --save-dim 5 --bandwidth-quantile 0.10 \
    --out-dir figures/ea_phase2_v8
