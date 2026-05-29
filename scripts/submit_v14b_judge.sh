#!/bin/bash
#SBATCH --job-name=v14b-judge-qwen
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/v14b-judge-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/v14b-judge-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
export HF_HOME=/data/artifacts/hf_cache
export HF_HUB_CACHE=/data/artifacts/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=/home/rkathuria/santi/.venv/bin/python

INPUT=${1:-/home/rkathuria/manifolds/figures/ea_phase2_v13_long/cot_rollouts.json}

cd "$PROJ"
PYTHONPATH=$PROJ/analysis:/home/rkathuria/santi:/home/rkathuria/santi/scripts \
    $PY scripts/v14b_judge_qwen.py \
    --input "$INPUT" \
    --judge-model /data/artifacts/rohan/santi/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
    --tp 1 --max-new-tokens 400 --gpu-mem 0.92
