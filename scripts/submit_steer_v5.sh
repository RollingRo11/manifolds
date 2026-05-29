#!/bin/bash
#SBATCH --job-name=steer-v5
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/steer-v5-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/steer-v5-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis $PY scripts/steer_v5.py \
    --steer-layers 19,28 \
    --n-waypoints 50 \
    --families weekday,temperature,age,year,color \
    --out-dir figures/steer_v5
