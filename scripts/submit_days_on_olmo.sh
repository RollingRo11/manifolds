#!/bin/bash
#SBATCH --job-name=days-olmo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/days-olmo-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/days-olmo-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis $PY scripts/days_on_olmo.py \
    --steer-layer 40 --n-waypoints 21 \
    --source Monday --target Friday
