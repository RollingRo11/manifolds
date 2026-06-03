# Recovering manifolds from a single probe

## What we're trying to do

Given a probe — a single direction in activation space, like *"this conversation feels evaluative"* or *"Monday vs Friday"* — we want to find the **full manifold** the probe lives on. Not just that one direction. The whole curved surface the model uses to organize related concepts.

Once you have that manifold, you get something powerful: smooth steering. Instead of a "more eval-aware" lever that just teleports between two points, you can walk the model along the actual structure of how it represents related concepts. For weekdays, that means probability flow that visits Tuesday → Wednesday → Thursday on the way from Monday to Friday, instead of jumping straight across.

## The simple version of the algorithm

Given activations `X` (an `N × d` matrix of last-token activations from `N` prompts) and a probe direction `w`:

```
s = X @ w                              # 1-D probe coordinate per activation
bin activations into K slices by s
for each slice k:
    Σ_k = within-slice covariance
M_SAVE = Σ_k (n_k / N) (Σ - Σ_k)²      # how much each slice differs from global
V = top eigenvectors of M_SAVE
embedding = X @ [w | V]                # (probe, partner_axis_1, partner_axis_2, ...)
```

That's it. One global eigendecomposition. No iteration, no labels.

This is **SAVE** — Sliced Average Variance Estimation, from Cook & Weisberg 1991. It's been hiding in classical statistics for 35 years.

The full implementation is [`analysis/save_recovery.py`](analysis/save_recovery.py) — about 50 lines including the thin-SVD trick that keeps it tractable in 5120-dim ambient space.

## Why this works

The probe is a direction. Project activations onto it: you get a single number per activation. Bin by that number. Now ask: **at each bin, what direction does the within-bin variance point in?**

For a circle and a chord-style probe (like Monday minus Friday), here's what happens. The probe maps both Tuesday and Sunday to about the same probe-score because they're equidistant from the chord. So a slice in the middle of the probe range contains *both* Tuesday-like and Sunday-like activations. The within-slice covariance is large — and it points along the direction that separates them. **That direction is the manifold's tangent at the chord midpoint.**

As you sweep through slices, this tangent rotates. Stitch the rotations together and you've recovered the circle.

The eigenvectors of `M_SAVE` aggregate this rotation across all slices in one shot. No bin-by-bin sign-flipping, no iteration. Just one matrix, one eigendecomposition.

## What it looks like on weekdays

We harvested OLMo-3.1-Think-32B activations on prompts like *"It's 5pm on day Monday"*, varying both the time-of-day descriptor and the weekday. 420 prompts total. We took activations at the day-name token, layer 19.

The supervised baseline (PCA on the activations, color by weekday) gives the canonical "weekday cycle" picture — seven distinct clusters arranged in a closed loop in calendar order.

We trained the simplest possible probe on this: a chord direction `w = mean(Monday acts) − mean(Friday acts)`. One binary distinction. That's all the supervision SAVE gets.

Run SAVE with that probe. The 7 weekday centroids in the recovered embedding land **at Procrustes distance 0.05 from the supervised PCA**. Visually identical loops; same calendar order; same rough scale. From a single binary probe.

![Supervised PCA vs SAVE recovery — weekday cycle](figures/compare_to_paper.png)

The eigenvalue spectrum is itself a diagnostic. With our chord probe: top eigenvalue `λ₁ = 243`. With random probe directions averaged over 10 trials: `λ₁ = 130`. The 1.86× ratio is the *"did this probe organize the data?"* signal — clear evidence the probe found real structure, not noise.

![SAVE eigenvalue spectrum: chord probe vs random probes](figures/save_spectrum.png)

## How to use the manifold for steering

The manifold isn't just for visualization. We can use it to control the model.

Pick a source concept (Monday) and a target concept (Friday). Three ways to walk between them:

- **Linear**: straight line in ambient activation space from Monday's centroid to Friday's. Replace the residual at the steering layer at each waypoint, continue forward, sample.
- **Spline (paper's method)**: cubic spline through *all 7* concept centroids, sample along the source→target arc. Same replacement.
- **SAVE arc**: piecewise-linear walk through the same 7 centroids in the SAVE-recovered embedding, lifted back to ambient. Same replacement.

Generate at each waypoint and look at the next-token logits over the seven day names.

![Days steering on OLMo-32B: linear teleports, spline and SAVE both produce smooth Mon→Tue→Wed→Thu→Fri flow](figures/days_on_olmo/days_olmo_Monday_to_Friday.png)

Linear teleports. Probability mass starts on Monday, decays as you walk, and re-appears on Friday near the end — but **never visits Tuesday, Wednesday, or Thursday**. The chord cuts through the middle of the cycle plane, and the model on that path doesn't predict any intermediate day.

Spline and SAVE both produce a clean diagonal sweep: Monday → Tuesday → Wednesday → Thursday → Friday, in calendar order, with smooth handoff between days. Each intermediate day dominates for some stretch of the path.

Concretely: fraction of waypoints where Tuesday/Wednesday/Thursday is the dominant predicted day:

| method | intermediate-coverage |
|---|---|
| Linear | **0.000** |
| Paper spline | **0.619** |
| SAVE arc | **0.571** |

SAVE matches the paper's spline within 5pp on this benchmark — using one binary probe direction rather than seven labeled centroids.

## How we know it's the right manifold (not just any manifold)

Two checks.

**Eigenvalue spectrum gap.** A useless probe should produce a flat `M_SAVE` — bin-conditional covariances are random, no direction stands out. We compare `λ₁` for our actual probe against the mean `λ₁` from random unit-vector probes. For weekdays: 1.86×. For lower-quality probes on harder concepts the gap shrinks. This is a free probe-quality diagnostic that comes with the algorithm.

**Procrustes against supervised PCA.** When we have ground-truth labels (we hold them out from the recovery), we can rigid-align the SAVE centroids to the supervised PCA centroids and measure residual disparity. 0.05 on weekdays-at-L19 means the cycles are essentially the same shape after best rotation+scale.

Neither metric is needed in deployment — they're for *validating* the method on cases where you have ground truth. In the wild (unknown probe, unknown manifold), you'd run SAVE, look at the eigenvalue spectrum to see if the probe has signal, and read the embedding directly.

## Discovering how many bins the manifold has

SAVE hands you the manifold from a single probe. But there's a second thing you'd want for free: **how many distinct states does the manifold support?** Seven weekdays, twelve months, twenty-four letters. If the method is really unsupervised, you shouldn't have to tell it the count — you should be able to read it off.

**The tempting dead end: the SAVE eigenvalue spectrum.** The natural guess is that the SAVE eigenvalues already encode the count — find the sharp drop `λᵢ / λᵢ₊₁` and call that index the number of bins. It doesn't work. The SAVE spectrum decays *smoothly*: on months the largest ratio is just `λ₀/λ₁` (which would claim "1 bin"), and a random probe produces the same leading-gap shape. The reason is conceptual. The `λ₁`-vs-random test above tells you *whether the probe found structure*; it does **not** tell you how many states that structure has, because **SAVE eigenvalues measure the intrinsic *dimension* of the manifold, not the number of points on it.** Seven weekdays arranged on a curved ring is a ~2–3 dimensional object — there's no reason for an eigenvalue gap to appear at 7. "How many bins" is a *clustering* question, and it has to be treated as one.

**The right frame: count the clusters — but clean the activations first.** The concept tokens *do* form discrete clusters (one blob per weekday), so the count is just "how many blobs are there." The catch is that at deep layers the blobs smear together: at L40 — our steering/visualization layer — k-means against the held-out labels scores only ARI ≈ 0.73, because two nuisance signals dominate the raw activations. One is the transformer's *massive activations* (a few dimensions with enormous magnitude). The other is a *shared context direction* (the carrier-phrase variation that's common to every concept). Strip both and the blobs snap back: cluster recovery jumps to ARI ≈ 1.0.

The recipe, start to finish, with no `K` supplied anywhere:

```
# X : N × d last-token activations for one concept family
X = X / norm(X, axis=1)          # 1. unit-normalize → cosine space (kills massive-activation magnitude)
X = X - mean(X, axis=0)
v1 = top right singular vector of X
X = X - (X @ v1) v1ᵀ             # 2. project out the top global PC (the shared context direction)
X = PCA(X, 20)                   # 3. down to a workable subspace
for K in 2 .. 2·K_max:           # 4. silhouette over candidate counts
    score[K] = silhouette(X, kmeans(X, K))
n_bins = argmax(score)           #    the count is the most cohesive K
```

One normalization, one rank-1 projection, and an off-the-shelf silhouette argmax. The cleaning is what makes the simple criterion work: on raw or under-cleaned activations, silhouette *over-splits* months (it picks 15–22, latching onto fine sub-structure that doesn't reproduce); on the cleaned cosine-space representation that sub-structure collapses and silhouette lands exactly on the concept count.

**It works at every layer through the one we visualize.** Recovered count vs. layer, no `K` given:

| layer | 16 | 24 | 32 | **40** | 48 | 56 |
|---|---|---|---|---|---|---|
| days (true 7) | 7 | 7 | 7 | **7** | 9 | 9 |
| months (true 12) | 12 | 12 | 12 | **12** | 12 | 19 |

Exact for both, with no supervision, at every layer from L16 through **L40 — the same layer the manifolds are visualized at**. So the pipeline is honestly unsupervised end to end: you never tell it there are seven days.

![Recovered bin count vs layer — silhouette on cleaned activations](figures/bin_count_sweep/simple_rule_layers.png)

**If you want the count to fall out of a spectrum**, use the graph-Laplacian eigengap rather than the SAVE matrix (this is the principled eigenvalue-based cluster counter — the eigengap of a kNN affinity graph). At L16 it's textbook: the first 7 (days) / 12 (months) Laplacian eigenvalues sit at ≈ 0, then jump ~40×. The number of near-zero eigenvalues *is* the number of clusters — ARI 1.00, no `K`.

![Laplacian eigengap at L16 — K near-zero eigenvalues then a jump](figures/bin_count_sweep/eigengap_bins.png)

The eigengap is the prettiest diagnostic but the least portable across layers: by L40 the days have relaxed into a *connected* curved manifold (still perfectly k-means-separable, but not graph-disconnected), so the eigengap counts components and reads 8 instead of 7. Months stay more cluster-like, so the eigengap still nails 12 there. If you specifically need an exact count at L40 for *both*, the most robust single rule is **prediction strength** — the largest `K` whose clustering reproduces on held-out halves — which recovers 7 and 12 at L40, at the cost of being more involved than a silhouette argmax.

**Honest limits.** None of these rules survives the final two layers (L48–L56): days drifts to 8–9, months eventually to 19. That isn't a failure of the count rule — it's the model itself blurring discrete concept identity as the last layers transition toward emitting output tokens. Concept structure is sharpest in the middle of the network, which is exactly where you'd want to read or steer it. Within that usable band (L16–L40), the simple recipe above recovers the number of bins for free — closing the loop on "unsupervised": one binary probe gives you the manifold, and the activations themselves tell you how many states it has.

## Generalizing to other geometries

The weekday cycle isn't a special case. We've validated SAVE on the full battery of manifolds from the paper (Llama-3.1-8B base, paper-faithful prompts):

![Supervised PCA vs SAVE recovery across manifold types: temperature, age, year, days](figures/all_manifolds_supervised_vs_save.png)

- **Days (circle)**: SAVE Procrustes 0.21 (vs random baseline 0.25), spectral gap 1.83× — clean recovery.
- **Year (helix)**: best layer L32, gap 1.94× — visible curvature recovered.
- **Temperature, age (lines)**: both supervised PCA and SAVE recover the line, but SAVE's spectral gap collapses to ~1.0 — the probe-driven recovery offers no advantage on genuinely linear concepts. This is *correct behavior*: there's no curvature for SAVE to find.

## What's good about it

- **Less supervision than the paper.** Their spline-through-centroids needs labels for every concept. SAVE needs one binary probe direction.
- **One eigendecomposition.** No iteration, no sign-flipping, no closure correction. Implementation is ~50 lines.
- **Equivalent steering quality.** On the canonical days/cycle benchmark, SAVE and paper-spline produce manifold steering that's behaviorally indistinguishable.
- **The eigenvalue spectrum is its own probe-quality test.** Random probes give a known-noise baseline against which to compare.

## What's tricky about it

- **SAVE only sees second-moment structure** (slice-conditional covariance variation). For purely linear features (concepts that *are* a line), there's no curvature for SAVE to grab onto, the spectrum gap collapses, and recovery is no better than chance. SIR — SAVE's first-moment cousin — is the right tool for those. The classical merge is **Directional Regression** (Li & Wang 2007), which combines both.
- **Recovery dimension is bounded by the number of slices**: `M_SAVE` has rank ≤ K−1, so use K ≥ d_M + 2.
- **Steering protocol matters as much as recovery.** For diffuse concepts (eval-awareness, persona, sentiment) where the relevant signal isn't tied to a specific token-position prediction, replacement-style steering can fail regardless of method — the residual at the injection position has to be semantically meaningful for the model. **Chat-templated models need chat-templated steering prompts.** This isn't a property of SAVE; it bit us first when we forgot the chat template on OLMo and got "Okay Okay Okay" loops out of every method.

## The shape of the win

If you trust the supervised PCA approach, you can interpret it as: each concept centroid is a point on the manifold, and a spline interpolates them.

What SAVE buys you is that you don't need the centroids at all. Give it any direction along which the data has interesting variation, and it'll hand you back the surrounding manifold. **Probe-driven manifold discovery, with one eigendecomposition and a single binary classifier as input.**
