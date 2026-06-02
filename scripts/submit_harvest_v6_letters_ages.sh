#!/bin/bash
#SBATCH --job-name=manifold-v6-letters-ages
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/home/rkathuria/manifolds/logs/harvest-v6-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/harvest-v6-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
PROJ=/home/rkathuria/manifolds
OUT=/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v6_letters_ages
mkdir -p "$PROJ/logs" "$OUT"

PY=$REPO/.venv/bin/python

export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub

cd "$REPO"

# Layers/model match concepts_olmo31_32b_v3 (days+months) for direct comparability.
"$PY" scripts/extract_sglang.py \
    --model allenai/Olmo-3.1-32B-Think \
    --layers 16,24,32,40,48,56 \
    --prompts $PROJ/data/prompts_v6_letters_ages.jsonl \
    --completions $PROJ/data/completions_v6_letters_ages.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 64 \
    --tp-size 2 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32

"$PY" scripts/verify.py "$OUT" || true
echo "HARVEST_DONE $OUT"
