"""Phase 9 — FORTRESS-derived behavior-contrastive EA direction + WildChat structure check.

FORTRESS provides antipodal pairs: same prompt, different rollouts, one labeled
"aware" and one "not aware". The mean difference is a PURE model-internal EA
direction (same prompt content, only behavior differs). Compare with:
  - testing_vs_conv (existing external probe at L40)
  - hua_et_al (existing external probe at L40)
  - our FS chord (the hand-crafted axis we've been using)

If FORTRESS-derived w aligns with testing_vs_conv but is orthogonal to FS,
that's strong evidence that our FS is measuring text-pattern features, not the
model's internal EA representation.

Then run SAVE with w_fortress on the FORTRESS data, see eigenvalue spectrum.

Phase B (WildChat) will run after 14712 lands; that section is at the bottom
guarded by a file-exists check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from load_acts_multi import read_last_token_multi_rollout
from save_recovery import save_recovery

LAYER = 40
FORTRESS_SHARD = Path("/data/artifacts/rohan/activation-harvester/fortress_balanced_v1/shard_dp00")
FORTRESS_LABELS = Path("/home/rkathuria/santi/data/probe_train/rollout_labels_fortress_balanced.jsonl")
EA_SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
EA_LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
WILDCHAT_SHARD = Path("/data/artifacts/rohan/manifolds/wildchat_olmo31_32b_v1/shard_dp00")
WILDCHAT_PROMPTS = Path("/home/rkathuria/manifolds/data/wildchat_prompts.jsonl")
PROBE_DIR = Path("/home/rkathuria/santi/logs/steering/per_layer_steer_files")
OUT_DIR = Path("/home/rkathuria/manifolds/figures/v9_fortress")


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def load_fortress_aware_labels():
    """Returns dict {(pid, cidx): aware:bool}."""
    out = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            out[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    print(f"=== PHASE A: FORTRESS antipodal-pair contrastive EA ===\n")
    print(f"loading FORTRESS L{LAYER} activations (multi-rollout)...")
    X_f, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = load_fortress_aware_labels()
    print(f"  {X_f.shape[0]} activations, {len(aware_map)} labels")

    # Match
    matched = [(i, aware_map[k]) for i, k in enumerate(keys) if k in aware_map]
    if not matched:
        print("[error] no matched aware labels — check pid/cidx alignment")
        return
    idx = np.array([m[0] for m in matched])
    aware = np.array([m[1] for m in matched])
    Xa = X_f[idx].astype(np.float64)
    print(f"  {len(idx)} matched   |   aware={aware.sum()}   non-aware={(~aware).sum()}")

    mean_aware = Xa[aware].mean(0)
    mean_non = Xa[~aware].mean(0)
    w_fortress = normalize(mean_aware - mean_non)
    chord_norm = float(np.linalg.norm(mean_aware - mean_non))
    print(f"  ‖mean_aware − mean_non_aware‖ = {chord_norm:.2f}  (FORTRESS contrast)")

    # External probes
    w_tvc = normalize(load_file(str(PROBE_DIR / f"testing_vs_conversation_opus46_L{LAYER}.safetensors"))[f"layer_{LAYER}"].numpy().astype(np.float64))
    w_hua = normalize(load_file(str(PROBE_DIR / f"hua_et_al_L{LAYER}.safetensors"))[f"layer_{LAYER}"].numpy().astype(np.float64))

    # FS chord (from EA shard)
    print(f"\nloading EA L{LAYER} activations + FS labels for cross-comparison...")
    X_ea, pids = read_last_token_per_seq(EA_SHARD, LAYER, position_from_end=10)
    ea_labels = load_labels(EA_LABELS)
    fs = np.array([ea_labels[p]["frame_score"] for p in pids])
    cents = np.stack([X_ea[fs == k].astype(np.float64).mean(0) for k in range(7)])
    w_fs_chord = normalize(cents[6] - cents[0])

    # ===== KEY COMPARISON =====
    print(f"\n=== Cross-direction cosines (after FORTRESS contrast) ===")
    print(f"{'':24s}  {'tvc':>8s}  {'hua':>8s}  {'fortress':>8s}  {'fs_chord':>8s}")
    rows = [("testing_vs_conv", w_tvc), ("hua_et_al", w_hua),
            ("FORTRESS aware-nonaware", w_fortress), ("FS_chord (FS=6−FS=0)", w_fs_chord)]
    cols = [w_tvc, w_hua, w_fortress, w_fs_chord]
    for name, v in rows:
        s = "  ".join(f"{v @ c:+8.3f}" for c in cols)
        print(f"  {name:22s}  {s}")

    # ===== Project EA-FS data onto w_fortress to check FS correlation =====
    print(f"\n=== Apply FORTRESS direction to EA-FS data ===")
    s_fortress_on_ea = X_ea.astype(np.float64) @ w_fortress
    rho = np.corrcoef(s_fortress_on_ea, fs)[0, 1]
    print(f"  corr(FORTRESS_score, FS_label) = {rho:+.3f}")
    print("  per-FS mean FORTRESS score: " +
          " ".join(f"FS{k}={s_fortress_on_ea[fs == k].mean():+.2f}" for k in range(7)))

    # ===== SAVE on FORTRESS data with w_fortress probe =====
    print(f"\n=== SAVE on FORTRESS using behavior-contrastive probe ===")
    save_res = save_recovery(Xa, w_fortress, n_slices=10, d_M=5)
    print(f"  top-5 eigvals: {save_res.eigvals[:5]}")
    print(f"  eigval[0]/eigval[20] = {save_res.eigvals[0] / max(save_res.eigvals[20], 1e-12):.2f}")
    print(f"  prob 1-D structure (eigval ratio if >5: yes, if <2: no)")

    np.savez(OUT_DIR / "fortress_directions.npz",
              w_fortress=w_fortress, w_tvc=w_tvc, w_hua=w_hua, w_fs_chord=w_fs_chord,
              chord_norm=chord_norm,
              save_eigvals=save_res.eigvals, save_dirs=save_res.save_dirs)

    # ===================================================================
    print(f"\n\n=== PHASE B: WildChat unsupervised ===\n")
    if not WILDCHAT_SHARD.exists():
        print(f"  WildChat shard not yet harvested ({WILDCHAT_SHARD} missing).")
        print(f"  Run after job 14712 finishes; this script will pick it up.")
        return

    print(f"loading WildChat L{LAYER}...")
    X_wc, wc_pids = read_last_token_per_seq(WILDCHAT_SHARD, LAYER, position_from_end=4)
    print(f"  {X_wc.shape[0]} activations")
    Xwf = X_wc.astype(np.float64)

    # Project onto each probe
    print(f"\n=== Probe-score distributions on WildChat (no labels) ===")
    for name, v in [("testing_vs_conv", w_tvc),
                     ("hua_et_al", w_hua),
                     ("FORTRESS aware-non", w_fortress),
                     ("FS chord", w_fs_chord)]:
        s = Xwf @ v
        q = np.quantile(s, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"  {name:22s}: 5/25/50/75/95% = {q[0]:+.2f} {q[1]:+.2f} {q[2]:+.2f} {q[3]:+.2f} {q[4]:+.2f}  std={s.std():.2f}")

    # Show extremes for testing_vs_conv
    print(f"\n=== top-5 / bottom-5 WildChat prompts by testing_vs_conv score ===")
    prompts_map = {}
    with open(WILDCHAT_PROMPTS) as f:
        for line in f:
            d = json.loads(line)
            prompts_map[d["id"]] = d["prompt"]
    s_tvc = Xwf @ w_tvc
    sort_idx = np.argsort(s_tvc)
    print("--- LOWEST testing_vs_conv (least eval-coded) ---")
    for i in sort_idx[:5]:
        pid = wc_pids[i]
        p = prompts_map.get(pid, "(unknown)")[:120]
        print(f"  s={s_tvc[i]:+.3f}  pid={pid}: {p!r}")
    print("--- HIGHEST testing_vs_conv (most eval-coded) ---")
    for i in sort_idx[-5:]:
        pid = wc_pids[i]
        p = prompts_map.get(pid, "(unknown)")[:120]
        print(f"  s={s_tvc[i]:+.3f}  pid={pid}: {p!r}")

    # SAVE on WildChat with testing_vs_conv probe
    print(f"\n=== SAVE on WildChat using testing_vs_conv probe ===")
    save_wc = save_recovery(Xwf, w_tvc, n_slices=10, d_M=5)
    print(f"  top-5 eigvals: {save_wc.eigvals[:5]}")
    print(f"  eigval[0]/eigval[20] = {save_wc.eigvals[0] / max(save_wc.eigvals[20], 1e-12):.2f}")

    # Compare top SAVE direction to existing probes
    save_dir0 = normalize(save_wc.save_dirs[:, 0])
    print(f"  cos(SAVE_dir0, testing_vs_conv) = {save_dir0 @ w_tvc:+.3f}")
    print(f"  cos(SAVE_dir0, hua_et_al)        = {save_dir0 @ w_hua:+.3f}")
    print(f"  cos(SAVE_dir0, FORTRESS)         = {save_dir0 @ w_fortress:+.3f}")
    print(f"  cos(SAVE_dir0, FS_chord)         = {save_dir0 @ w_fs_chord:+.3f}")

    np.savez(OUT_DIR / "wildchat_save.npz",
              save_eigvals=save_wc.eigvals,
              save_dirs=save_wc.save_dirs,
              probe_scores=Xwf @ w_tvc, hua_scores=Xwf @ w_hua,
              fortress_scores=Xwf @ w_fortress, fs_chord_scores=Xwf @ w_fs_chord,
              pids=np.array(wc_pids))


if __name__ == "__main__":
    main()
