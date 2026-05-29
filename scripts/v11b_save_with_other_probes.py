"""SAVE with multiple probe vectors on the same FORTRESS data.

Each probe = different 1-D slicing → different SAVE matrix → different top eigvecs.
We compare:
  - probe = w_fortress (aware-vs-non-aware mean-diff): the 'native' contrast
  - probe = testing_vs_conv: external Apollo probe
  - probe = hua_et_al: external Hua probe
  - probe = w_FS_chord (our hand-labeled FS=6 - FS=0)

For each probe, run SAVE with K=10 quantile slices on the FORTRESS activations.
Report:
  - Top eigenvalue ratio (signal vs noise floor)
  - Top SAVE eigvec direction
  - Class-discrimination quality of top eigvec on aware/non-aware labels
  - Cosines between top eigvecs across probes (do they recover same structure?)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file as np_load

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from load_acts_multi import read_last_token_multi_rollout
from save_recovery import save_recovery

LAYER = 40
FORTRESS_SHARD = Path("/data/artifacts/rohan/activation-harvester/fortress_balanced_v1/shard_dp00")
FORTRESS_LABELS = Path("/home/rkathuria/santi/data/probe_train/rollout_labels_fortress_balanced.jsonl")
EA_SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
EA_LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
PROBE_DIR = Path("/home/rkathuria/santi/logs/steering/per_layer_steer_files")
OUT_DIR = Path("/home/rkathuria/manifolds/figures/v11b_other_probes")


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== loading FORTRESS L40 + labels ===")
    X, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            aware_map[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])
    matched = [(i, aware_map[k]) for i, k in enumerate(keys) if k in aware_map]
    idx = np.array([m[0] for m in matched])
    aware = np.array([m[1] for m in matched])
    Xa = X[idx].astype(np.float64)
    print(f"  {Xa.shape[0]} matched acts (aware={aware.sum()} non={(~aware).sum()})")

    # Build the 4 probe directions
    mean_a = Xa[aware].mean(0); mean_n = Xa[~aware].mean(0)
    w_fortress = normalize(mean_a - mean_n)

    w_tvc = normalize(np_load(str(PROBE_DIR / f"testing_vs_conversation_opus46_L{LAYER}.safetensors"))[f"layer_{LAYER}"].astype(np.float64))
    w_hua = normalize(np_load(str(PROBE_DIR / f"hua_et_al_L{LAYER}.safetensors"))[f"layer_{LAYER}"].astype(np.float64))

    print("\nloading EA L40 for FS chord direction...")
    X_ea, pids = read_last_token_per_seq(EA_SHARD, LAYER, position_from_end=10)
    ea_labels = load_labels(EA_LABELS)
    fs = np.array([ea_labels[p]["frame_score"] for p in pids])
    cents_ea = np.stack([X_ea[fs == k].astype(np.float64).mean(0) for k in range(7)])
    w_fs = normalize(cents_ea[6] - cents_ea[0])

    probes = {
        "w_fortress (aware-non)": w_fortress,
        "testing_vs_conv": w_tvc,
        "hua_et_al": w_hua,
        "w_FS_chord": w_fs,
    }

    # ===== Run SAVE with each probe =====
    print("\n=== SAVE on FORTRESS with each probe (K=10 quantile slices) ===")
    results = {}
    for name, w in probes.items():
        res = save_recovery(Xa, w, n_slices=10, d_M=10)
        results[name] = res
        ratio = res.eigvals[0] / max(res.eigvals[20], 1e-12)
        print(f"\n  -- probe = {name} --")
        print(f"    top-3 eigvals: {res.eigvals[:3]}")
        print(f"    eigval[0]/eigval[20] = {ratio:.1f}  (manifold signal vs noise floor)")

        # Top eigvec direction quality
        v0 = res.save_dirs[:, 0]
        # Discrimination on aware/non-aware
        sa = Xa[aware] @ v0
        sn = Xa[~aware] @ v0
        d_score = (sa.mean() - sn.mean()) / np.sqrt(sa.var() + sn.var() + 1e-12)
        # Cosine to all probes
        cos_w = {pname: float(v0 @ pw) for pname, pw in probes.items()}
        print(f"    SAVE dir[0] ‖aware-mean − non-aware-mean‖_dir: {abs(sa.mean() - sn.mean()):.2f}")
        print(f"    SAVE dir[0] class d-score: {d_score:+.3f}")
        print(f"    cos(SAVE_dir0, probes): " +
               ", ".join(f"{p}={c:+.3f}" for p, c in cos_w.items()))

    # ===== Cross-comparison: are the top eigvecs the same? =====
    print("\n=== top SAVE eigvec cross-cosines (do probes find same structure?) ===")
    names = list(probes.keys())
    print(f"  {'':24s}  " + "  ".join(f"{n[:10]:>10s}" for n in names))
    for n1 in names:
        v1 = results[n1].save_dirs[:, 0]
        row = "  ".join(f"{abs(v1 @ results[n2].save_dirs[:, 0]):>10.3f}" for n2 in names)
        print(f"  {n1:22s}  {row}")

    print("\n=== top SAVE eigvec[1..2] cross-cosines (second/third-order structure) ===")
    for k in range(1, 3):
        print(f"\n  -- eigvec index {k} --")
        for n1 in names:
            v1 = results[n1].save_dirs[:, k]
            row = "  ".join(f"{abs(v1 @ results[n2].save_dirs[:, k]):>10.3f}" for n2 in names)
            print(f"  {n1:22s}  {row}")

    # Save data
    np.savez(OUT_DIR / "save_with_probes.npz",
              **{f"{n}_eigvals": r.eigvals for n, r in results.items()},
              **{f"{n}_save_dirs": r.save_dirs for n, r in results.items()})
    print(f"\nwrote {OUT_DIR}/save_with_probes.npz")


if __name__ == "__main__":
    main()
