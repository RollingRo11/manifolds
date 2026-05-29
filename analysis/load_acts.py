"""Load harvested activations into per-sequence arrays.

The harvester writes per-sequence chunks into one big binary file per layer with
an index.jsonl pointing at (start, n_tokens). For our concept-prediction setup,
we want exactly one vector per sequence: the *last* token of (prompt+completion),
which is the concept token.

read_concept_acts(shard_dir, layer) -> dict[pid] -> np.ndarray[d_model] (float32)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def _load_meta(shard_dir: Path) -> dict:
    p = shard_dir / "meta.json"
    txt = p.read_text() if p.exists() else ""
    if txt.strip():
        return json.loads(txt)
    # Reconstruct from index.jsonl if the writer didn't flush meta.
    layers, ntok = set(), {}
    with open(shard_dir / "index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            layers.add(r["layer"])
            ntok[r["layer"]] = ntok.get(r["layer"], 0) + r["n_tokens"]
    # d_model from any bin file size: bytes / (n_tokens * 2)  [bf16 default]
    L = next(iter(layers))
    bin_size = (shard_dir / f"acts_layer_{L:03d}.bin").stat().st_size
    d_model = bin_size // (ntok[L] * 2)
    return {"model": "(unknown)", "layers": sorted(layers),
            "d_model": int(d_model), "dtype": "bfloat16", "bytes_per_elem": 2,
            "has_scales": False}


def _load_index(shard_dir: Path, layer: int) -> list[dict]:
    rows = []
    with open(shard_dir / "index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["layer"] == layer:
                rows.append(r)
    return rows


def read_last_token_per_seq(shard_dir: Path, layer: int,
                            position_from_end: int = 0) -> tuple[np.ndarray, list[int]]:
    """Return (X[N, d_model] float32, pids[N]) — activation at position
    (last_token - position_from_end) per sequence. Default 0 = the last token.
    Use position_from_end=k to read the k-th-from-last token (e.g., to skip
    over a fixed-length completion suffix).
    """
    shard_dir = Path(shard_dir)
    meta = _load_meta(shard_dir)
    d = meta["d_model"]
    dtype = meta.get("dtype", "bfloat16")
    bpe = meta.get("bytes_per_elem", 2)
    if dtype != "bfloat16":
        raise NotImplementedError(f"only bf16 implemented; got {dtype}")

    rows = sorted(_load_index(shard_dir, layer), key=lambda r: r["prompt_id"])
    bin_path = shard_dir / f"acts_layer_{layer:03d}.bin"
    out = np.empty((len(rows), d), dtype=np.float32)
    pids: list[int] = []
    with open(bin_path, "rb") as f:
        for i, r in enumerate(rows):
            target_offset = r["start"] + r["n_tokens"] - 1 - position_from_end
            if target_offset < r["start"]:
                target_offset = r["start"]  # clamp
            f.seek(target_offset * d * bpe)
            raw = f.read(d * bpe)
            arr = np.frombuffer(raw, dtype=bfloat16).astype(np.float32)
            out[i] = arr
            pids.append(r["prompt_id"])
    return out, pids


def load_labels(labels_jsonl: Path) -> dict[int, dict]:
    out = {}
    with open(labels_jsonl) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = d
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--layer", type=int, default=32)
    args = ap.parse_args()
    X, pids = read_last_token_per_seq(Path(args.shard_dir), args.layer)
    print(f"loaded layer {args.layer}: X.shape={X.shape}  pids[:5]={pids[:5]}")
    print(f"norm stats: mean={np.linalg.norm(X, axis=1).mean():.2f} "
          f"min={np.linalg.norm(X, axis=1).min():.2f} "
          f"max={np.linalg.norm(X, axis=1).max():.2f}")
