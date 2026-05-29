#!/bin/bash
#SBATCH --job-name=harvest-wildchat
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/harvest-wildchat-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/harvest-wildchat-%j.err

set -euo pipefail
REPO=/home/rkathuria/activation-harvester
OUT=/data/artifacts/rohan/manifolds/wildchat_olmo31_32b_v1
mkdir -p "$OUT"

export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub
export PYTHONUNBUFFERED=1
PY=$REPO/.venv/bin/python

cd "$REPO"
"$PY" scripts/extract_sglang.py \
    --model /data/artifacts/rohan/santi/hf_cache/hub/models--allenai--Olmo-3.1-32B-Think/snapshots/$(ls /data/artifacts/rohan/santi/hf_cache/hub/models--allenai--Olmo-3.1-32B-Think/snapshots | head -1) \
    --layers 16,24,32,40,48 \
    --prompts /home/rkathuria/manifolds/data/wildchat_prompts.jsonl \
    --completions /home/rkathuria/manifolds/data/wildchat_completions.jsonl.zst \
    --output-dir "$OUT" \
    --tp-size 1 \
    --dtype bfloat16 \
    --chunked-prefill-size 32768 \
    --max-running-requests 8 \
    --mem-fraction-static 0.85
