#!/bin/bash
#SBATCH --job-name=ea-v21-fortress
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/ea-v21-fortress-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/ea-v21-fortress-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
export HF_DATASETS_CACHE=/data/artifacts/hf_cache
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/activation-harvester/src \
    $PY scripts/ea_phase2_v21_fortress_scaled.py \
    --n-prompts 100 --n-rollouts 25 --max-new-tokens 300 \
    --magnitude 5.0 --target-quantile 0.85 --max-push-mult 2.0 \
    --tp-size 1 --mem-fraction-static 0.88 --max-running-requests 32 \
    --n-bins 21 --save-dim 3 --seed 42 \
    --out-dir figures/ea_phase2_v21_fortress
