#!/bin/bash
#SBATCH --job-name=v22-triple-judge
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/v22-judge-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/v22-judge-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/hf_cache
export HF_HUB_CACHE=/data/artifacts/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/santi/.venv/bin/python

INPUT=${1:-/home/rkathuria/manifolds/figures/ea_phase2_v21_fortress/cot_rollouts.json}

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/santi:/home/rkathuria/santi/scripts \
    $PY scripts/v22_triple_judge.py \
    --input "$INPUT" \
    --tp 1 --gpu-mem 0.92 --max-new-tokens 400
