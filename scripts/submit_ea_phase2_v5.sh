#!/bin/bash
#SBATCH --job-name=ea-phase2-v5
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/ea-phase2-v5-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/ea-phase2-v5-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
PY=/home/rkathuria/activation-harvester/.venv/bin/python

cd "$PROJ"
# Sweep alpha to find the strength where behavioral steering kicks in
for ALPHA in 3.0 5.0 8.0; do
    PYTHONPATH=$PROJ/analysis $PY scripts/ea_phase2_v5_allpos.py \
        --steer-layer 40 --n-waypoints 21 --alpha $ALPHA --position-from-end 10 \
        --out-dir figures/ea_phase2_v5_alpha${ALPHA}
done
