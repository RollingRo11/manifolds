#!/bin/bash
#SBATCH --job-name=manifold-steer
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/steer-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/steer-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub

cd "$PROJ"
PY=/home/rkathuria/activation-harvester/.venv/bin/python

PYTHONPATH=$PROJ/analysis $PY scripts/steer_save_vs_linear.py \
    --steer-layer 19 \
    --n-waypoints 50 \
    --source Monday --target Friday \
    --out figures/steer_L19_Mon_Fri.png
