#!/bin/bash
#SBATCH --job-name=manifold-steer-all
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/steer-all-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/steer-all-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis $PY scripts/steer_all_paper.py \
    --steer-layer 19 --n-waypoints 50 \
    --families weekday,temperature,age,year \
    --out-dir figures/steer_battery
