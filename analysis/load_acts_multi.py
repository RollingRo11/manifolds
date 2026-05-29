"""Multi-rollout aware loader: reads activations indexed by (prompt_id, completion_idx).

Existing load_acts.py assumes one rollout per prompt and sorts by prompt_id.
FORTRESS-style data has multiple rollouts per prompt with completion_idx labels,
so we need to preserve cidx in the index.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from load_acts import _load_meta


def read_last_token_multi_rollout(shard_dir: Path, layer: int,
                                    position_from_end: int = 0
                                    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (X[N, d_model] float32, [(pid, cidx)]) — last-token activations,
    one per (prompt_id, completion_idx) pair."""
    shard_dir = Path(shard_dir)
    meta = _load_meta(shard_dir)
    d = meta["d_model"]
    bpe = meta.get("bytes_per_elem", 2)

    rows = []
    with open(shard_dir / "index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["layer"] == layer:
                rows.append(r)
    rows.sort(key=lambda r: (r["prompt_id"], r.get("completion_idx", 0)))

    bin_path = shard_dir / f"acts_layer_{layer:03d}.bin"
    out = np.empty((len(rows), d), dtype=np.float32)
    keys: list[tuple[int, int]] = []
    with open(bin_path, "rb") as f:
        for i, r in enumerate(rows):
            target_offset = r["start"] + r["n_tokens"] - 1 - position_from_end
            if target_offset < r["start"]:
                target_offset = r["start"]
            f.seek(target_offset * d * bpe)
            raw = f.read(d * bpe)
            arr = np.frombuffer(raw, dtype=bfloat16).astype(np.float32)
            out[i] = arr
            keys.append((int(r["prompt_id"]), int(r.get("completion_idx", 0))))
    return out, keys
