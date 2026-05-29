"""Compare the testing_vs_conv probe direction to a probe trained on our FS labels.

If our hand-crafted FS axis aligns with what the model encodes, w_FS and w_probe
should be largely colinear. If not — we've been chasing different structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq

SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
LAYER = 40
PROBE_PATH = ("/home/rkathuria/santi/logs/steering/per_layer_steer_files/"
              "testing_vs_conversation_opus46_L40.safetensors")
HUA_PATH = ("/home/rkathuria/santi/logs/steering/per_layer_steer_files/"
            "hua_et_al_L40.safetensors")


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    print(f"=== loading EA acts at L{LAYER} ===")
    labels = load_labels(LABELS)
    X, pids = read_last_token_per_seq(SHARD, LAYER, position_from_end=10)
    fs = np.array([labels[p]["frame_score"] for p in pids])
    print(f"  {X.shape[0]} acts, FS dist: {np.bincount(fs)}")
    Xf = X.astype(np.float64)

    print("\n=== external probes ===")
    w_tvc = load_file(PROBE_PATH)[f"layer_{LAYER}"].cpu().numpy().astype(np.float64)
    w_hua = load_file(HUA_PATH)[f"layer_{LAYER}"].cpu().numpy().astype(np.float64)
    w_tvc = normalize(w_tvc); w_hua = normalize(w_hua)
    print(f"  cos(testing_vs_conv, hua_et_al) = {w_tvc @ w_hua:+.4f}  "
          f"(consistency check among external EA probes)")

    print("\n=== train FS-supervised directions ===")
    # 1. Centroid difference: c_FS6 - c_FS0
    cents = np.stack([Xf[fs == k].mean(0) for k in range(7)])
    w_fs_chord = normalize(cents[6] - cents[0])
    print(f"  w_FS_chord: ‖c_FS6 - c_FS0‖ = {np.linalg.norm(cents[6] - cents[0]):.2f}")

    # 2. Logistic regression FS=6 vs FS=0
    mask_06 = (fs == 0) | (fs == 6)
    y_06 = (fs[mask_06] == 6).astype(int)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(Xf[mask_06], y_06)
    w_fs_logit = normalize(lr.coef_[0])
    print(f"  w_FS_logit (FS=6 vs FS=0): train acc = {lr.score(Xf[mask_06], y_06):.3f}")

    # 3. Ridge regression on FS as continuous
    rr = Ridge(alpha=1.0)
    rr.fit(Xf, fs.astype(float))
    w_fs_ridge = normalize(rr.coef_)
    pred = rr.predict(Xf)
    rho = np.corrcoef(pred, fs)[0, 1]
    print(f"  w_FS_ridge (continuous regression): corr(pred, FS) = {rho:.3f}")

    print("\n=== cross-direction cosines ===")
    print(f"{'':24s}  {'tvc':>8s}  {'hua':>8s}  {'fs_chord':>8s}  {'fs_logit':>8s}  {'fs_ridge':>8s}")
    rows = [
        ("testing_vs_conv (tvc)", w_tvc),
        ("hua_et_al (hua)", w_hua),
        ("FS chord (c6-c0)", w_fs_chord),
        ("FS logit (c0/c6 LR)", w_fs_logit),
        ("FS ridge (continuous)", w_fs_ridge),
    ]
    cols = [w_tvc, w_hua, w_fs_chord, w_fs_logit, w_fs_ridge]
    for name, v in rows:
        s = "  ".join(f"{v @ c:+8.3f}" for c in cols)
        print(f"  {name:22s}  {s}")

    print("\n=== probe-score distributions across FS levels ===")
    for name, v in rows:
        scores = Xf @ v
        per_fs = [scores[fs == k] for k in range(7)]
        means = [np.mean(s) for s in per_fs]
        print(f"  {name:22s}: " + " ".join(f"FS{k}={m:+.2f}" for k, m in enumerate(means)))

    print("\n=== rank-1 probe-score correlation with FS ordinal ===")
    for name, v in rows:
        scores = Xf @ v
        rho = np.corrcoef(scores, fs)[0, 1]
        print(f"  {name:22s}: corr(score, FS) = {rho:+.3f}")


if __name__ == "__main__":
    main()
