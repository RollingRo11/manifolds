#!/bin/bash
#SBATCH --job-name=ea-phase2-v6
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/ea-phase2-v6-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/ea-phase2-v6-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
PYTHONPATH=$PROJ/analysis $PY scripts/ea_phase2_v6_alphasweep.py \
    --steer-layer 40 --n-waypoints 11 --position-from-end 10 \
    --n-rollouts 3 --max-new-tokens 80 \
    --alphas 1.0 2.0 3.0 5.0 7.0 10.0 \
    --out-dir figures/ea_phase2_v6
