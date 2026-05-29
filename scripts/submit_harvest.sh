#!/bin/bash
#SBATCH --job-name=manifold-harvest
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/rkathuria/manifolds/logs/harvest-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/harvest-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
PROJ=/home/rkathuria/manifolds
OUT=/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v1
mkdir -p "$PROJ/logs" "$OUT"

PY=$REPO/.venv/bin/python

# Use the cluster-wide HF cache that already has the 32B Think weights.
export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub

cd "$REPO"

"$PY" scripts/extract_sglang.py \
    --model allenai/Olmo-3.1-32B-Think \
    --layers 16,24,32,40,48,56 \
    --prompts $PROJ/data/prompts.jsonl \
    --completions $PROJ/data/completions.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 256 \
    --tp-size 2 \
    --chunked-prefill-size 16384 \
    --max-running-requests 32

"$PY" scripts/verify.py "$OUT"
