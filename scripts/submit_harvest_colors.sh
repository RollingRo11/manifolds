#!/bin/bash
#SBATCH --job-name=harvest-colors
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/harvest-colors-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/harvest-colors-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
PROJ=/home/rkathuria/manifolds
OUT=/data/artifacts/rohan/manifolds/colors_llama31_8b_base
mkdir -p "$PROJ/logs" "$OUT"

PY=$REPO/.venv/bin/python

export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub

cd "$REPO"

LLAMA_PATH=/data/artifacts/rohan/santi/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

"$PY" scripts/extract_sglang.py \
    --model "$LLAMA_PATH" \
    --layers 12,16,19,22,26 \
    --prompts $PROJ/data/prompts_colors.jsonl \
    --completions $PROJ/data/completions_colors.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 64 \
    --tp-size 1 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32

"$PY" scripts/verify.py "$OUT" || true
