#!/bin/bash
#SBATCH --job-name=manifold-llama-paper
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/harvest-llama-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/harvest-llama-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
PROJ=/home/rkathuria/manifolds
OUT=/data/artifacts/rohan/manifolds/paper_llama31_8b_base_v4
mkdir -p "$PROJ/logs" "$OUT"

PY=$REPO/.venv/bin/python

export HF_HOME=/data/artifacts/rohan/santi/hf_cache
export HF_HUB_CACHE=/data/artifacts/rohan/santi/hf_cache/hub

cd "$REPO"

# Concatenate the v3 weekday prompts and the v4 paper-faithful (temperature/age/year)
# into one harvest. Llama-3.1-8B has 32 layers; paper used L19. We sweep around it.
cat $PROJ/data/prompts_v3.jsonl $PROJ/data/prompts_v4_paper.jsonl > /tmp/llama_paper_prompts.jsonl
# Renumber ids so they're unique
"$PY" -c "
import json
rows = []
for line in open('/tmp/llama_paper_prompts.jsonl'):
    rows.append(json.loads(line))
with open('/tmp/llama_paper_prompts.jsonl', 'w') as f:
    for new_id, r in enumerate(rows):
        r['id'] = new_id
        f.write(json.dumps(r) + '\n')
print(f'merged {len(rows)} prompts')
"

# Build matching completions (need to merge labels too to know what to put in completions)
"$PY" -c "
import json, zstandard as zstd
labels = {}
for f in ['$PROJ/data/labels_v3.jsonl', '$PROJ/data/labels_v4_paper.jsonl']:
    for line in open(f):
        d = json.loads(line)
        labels[(d.get('family'), d.get('concept', None) or d.get('value', None))] = d
# Read the merged prompts and find the original label by content
v3_labels = {json.loads(l)['id']: json.loads(l) for l in open('$PROJ/data/labels_v3.jsonl')}
v4_labels = {json.loads(l)['id']: json.loads(l) for l in open('$PROJ/data/labels_v4_paper.jsonl')}
v3_n = max(v3_labels) + 1
out_labels = []
out_compls = []
for line in open('/tmp/llama_paper_prompts.jsonl'):
    p = json.loads(line)
    new_id = p['id']
    # If new_id < n_v3, it came from v3
    if new_id < v3_n:
        lbl = v3_labels[new_id]
        compl_text = ' ' + lbl['concept']
    else:
        lbl = v4_labels[new_id - v3_n]
        compl_text = ' ' + str(lbl['value'])
    lbl_out = dict(lbl); lbl_out['id'] = new_id
    out_labels.append(lbl_out)
    out_compls.append({'prompt_id': new_id, 'completion_idx': 0, 'text': compl_text})

with open('/tmp/llama_paper_labels.jsonl', 'w') as f:
    for l in out_labels:
        f.write(json.dumps(l) + '\n')

cctx = zstd.ZstdCompressor(level=3)
with open('/tmp/llama_paper_completions.jsonl.zst', 'wb') as f, cctx.stream_writer(f) as w:
    for c in out_compls:
        w.write((json.dumps(c) + '\n').encode())

print(f'wrote {len(out_labels)} labels, {len(out_compls)} completions')
"

cp /tmp/llama_paper_labels.jsonl $PROJ/data/labels_llama_paper.jsonl

# Use the snapshot path directly — paper's actual model
LLAMA_PATH=/data/artifacts/rohan/santi/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

"$PY" scripts/extract_sglang.py \
    --model "$LLAMA_PATH" \
    --layers 12,16,19,22,26 \
    --prompts /tmp/llama_paper_prompts.jsonl \
    --completions /tmp/llama_paper_completions.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 64 \
    --tp-size 1 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32

"$PY" scripts/verify.py "$OUT" || true
