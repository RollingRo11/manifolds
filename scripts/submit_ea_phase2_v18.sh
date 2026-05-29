#!/bin/bash
#SBATCH --job-name=ea-v18-spline
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/home/rkathuria/manifolds/logs/ea-v18-spline-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/ea-v18-spline-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/activation-harvester/src \
    $PY scripts/ea_phase2_v18_local_spline.py \
    --magnitudes 0.5 1.0 1.5 2.0 2.5 3.0 5.0 \
    --n-rollouts 5 --max-new-tokens 300 --tp-size 1 \
    --mem-fraction-static 0.88 --max-running-requests 16 \
    --knn-k 60 --n-bins 7 \
    --out-dir figures/ea_phase2_v18_local_spline
