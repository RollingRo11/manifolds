#!/bin/bash
#SBATCH --job-name=v14-judge
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/v14-judge-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/v14-judge-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/activation-harvester/.venv/bin/python

INPUT=${1:-/home/rkathuria/manifolds/figures/ea_phase2_v13_long/cot_rollouts.json}

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/activation-harvester/src \
    $PY scripts/v14_judge.py \
    --input "$INPUT" \
    --max-new-tokens 10 \
    --mem-fraction-static 0.88 \
    --max-running-requests 32
