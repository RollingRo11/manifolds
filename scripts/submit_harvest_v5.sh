#!/bin/bash
#SBATCH --job-name=harvest-v5
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/harvest-v5-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/harvest-v5-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
PROJ=/home/rkathuria/manifolds
OUT=/data/artifacts/rohan/manifolds/v5_llama31_8b_base
mkdir -p "$PROJ/logs" "$OUT"

PY=$REPO/.venv/bin/python
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
LLAMA=/data/artifacts/rohan/santi/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

cd "$REPO"

"$PY" scripts/extract_sglang.py \
    --model "$LLAMA" \
    --layers 16,19,22,26,28,30 \
    --prompts $PROJ/data/prompts_v5.jsonl \
    --completions $PROJ/data/completions_v5.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 64 \
    --tp-size 1 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32

"$PY" scripts/verify.py "$OUT" || true
