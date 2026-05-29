"""For each available contrastive probe, run v1 and v3 on the eval-awareness
shard and show 3D embeddings side by side.

Output: one PNG per probe with two 3D panels (v1, v3), colored by held-back
frame_score (0=casual → 6=explicit eval). The probe-recovery's job is to
reveal what the probe encodes; comparing v1 vs v3 shows whether the
recovered geometry is probe-driven (v1) or data-PCA-driven (v3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from sir_recovery import slice_and_local_pca
from sir_recovery_v3 import global_residual_recover


def load_eval_awareness():
    shard = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
    labels_p = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
    return shard, load_labels(labels_p)


def discover_probes(probes_dir: Path, layer: int) -> list[tuple[str, np.ndarray, str]]:
    """Returns [(probe_name, w, file_stem), ...] for all probes that have
    a key matching the requested layer. Skips dataset-derived probes
    (harmbench/fortress) for safety."""
    SKIP = ("harmbench", "fortress")
    out = []
    for f in probes_dir.glob("*.npz"):
        if any(s in f.name.lower() for s in SKIP):
            continue
        try:
            npz = np.load(f, allow_pickle=False)
        except Exception:
            continue
        for key in npz.files:
            arr = npz[key]
            # require name pattern "L{layer}_..."
            tag = f"L{layer}_"
            if not key.startswith(tag):
                continue
            if arr.ndim == 1 and arr.shape[0] == 5120:
                out.append((key.removeprefix(tag), arr, f.stem))
            elif arr.ndim == 2 and arr.shape[-1] == 5120:
                out.append((key.removeprefix(tag) + "[0]", arr[0], f.stem))
    return out


def plot_panel(ax, emb, frame_scores, title):
    cmap = plt.get_cmap("viridis")
    sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                     c=frame_scores, cmap=cmap, s=18, alpha=0.7,
                     vmin=0, vmax=6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("axis 1"); ax.set_ylabel("axis 2"); ax.set_zlabel("axis 3")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir", default="/data/artifacts/rohan/santi/probes")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--n-partner", type=int, default=2)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/probe_grid")
    args = ap.parse_args()

    shard, labels = load_eval_awareness()
    X, pids = read_last_token_per_seq(shard, args.layer)
    fs = np.array([labels[p]["frame_score"] for p in pids])
    print(f"loaded {X.shape[0]} eval-awareness activations at L{args.layer}")

    probes = discover_probes(Path(args.probes_dir), args.layer)
    print(f"found {len(probes)} probes")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # One figure per probe, two 3D panels
    for name, w, src in probes:
        v1 = slice_and_local_pca(X, w, n_bins=args.n_bins, n_partner=args.n_partner)
        v3 = global_residual_recover(X, w, n_partner=args.n_partner)

        fig = plt.figure(figsize=(14, 6))
        ax_l = fig.add_subplot(1, 2, 1, projection="3d")
        sc1 = plot_panel(ax_l, v1.embedding, fs, f"v1 per-bin SIR")
        ax_r = fig.add_subplot(1, 2, 2, projection="3d")
        sc2 = plot_panel(ax_r, v3, fs, f"v3 global residual-PCA")
        cb = fig.colorbar(sc1, ax=[ax_l, ax_r], shrink=0.7, label="frame_score (0=casual, 6=eval)")
        fig.suptitle(f"{src}:{name}  (layer {args.layer})\n"
                     f"data: eval-awareness shard (n={X.shape[0]})",
                     fontsize=12)
        out = out_dir / f"L{args.layer:03d}_{src}_{name}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  wrote {out.name}")

    print(f"\nAll figures in {out_dir}")


if __name__ == "__main__":
    main()
