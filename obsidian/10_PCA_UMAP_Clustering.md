# PCA & UMAP Clustering — Notes

Reference for the PCA/UMAP + KMeans phase-detection pipeline. Recaps the
clustering work done on the attractive-Hubbard pilot
(`integrated_pair_{position,momentum}`, `L=8`, `U=−4`, `n=0.875`,
FSS β_c = 8.5).

---

## What the pipeline does (short version)

1. Run **linear PCA** on the centered correlator data — never modified.
2. Run **UMAP** to a 2-D embedding (separate path).
3. Cluster (k=2) with KMeans on top of either embedding.
4. Identify the ordered phase as the cluster with higher ⟨β⟩.
5. Compute `f_KVBS(β)` = fraction of samples at β assigned to the ordered
   cluster; fit a sigmoid for `β_c`, peak of `df_KVBS/dβ` for a second
   estimate.

The clustering step is what we tuned in this session — PCA itself was
restored to a plain `KMeans(P[:, :2])` call by request.

---

## What changed in the codebase

| Where | Change |
|---|---|
| `ssh_ml_phase/pca_method.py` | New `_cluster_pca_scores(P, method)` and `cluster_method` parameter on `extract_beta_c`, `extract_alpha_c`, `run_all`, `run_all_alpha_sweep`. Default `"raw_2d"` (canonical linear behaviour, unchanged). Options: `"spherical"`, `"arcsinh"`. |
| `ssh_ml_phase/umap_method.py` | New `scale_kmeans` parameter (default `True`) on the same set of functions. Z-scores the 2-D embedding before KMeans. Top-of-module note explains z-scoring. |
| `ssh_ml_phase/pipeline.py` | Exposes `pca_cluster_method` and `umap_scale_kmeans` to `pipeline.run(...)`. |
| `plot_pca_diagnostics.py` | Multi-page PDF: P_k vs β with mean inset; means overlay (std units); susceptibility overlay (Var/λ_k); eigenvalue spectrum; (P1, P2) scatter coloured by β with `RdBu` diverging at β_c; PC1 loading on lattice/BZ (sign-pinned); per-β P1 ridgeline; skew/kurtosis to stdout. |
| `plot_pca_clustering_comparison.py` | One-page side-by-side comparison of all four PCA clustering recipes for both observables. |

---

## Diagnostics worth keeping in mind

| Quantity | What it tells you | Where to read it |
|---|---|---|
| Eigenvalue spectrum | If λ1 dominates and λ1/λ2 ≫ 1, the data is effectively 1-D and clustering on PC2 is fighting noise. | `pca_diagnostics` page 1, row 5. |
| ⟨P_k⟩_β / σ_k | Per-PC mean **trajectory** in dimensionless units. Picks up the order-parameter ramp. | row 3. |
| Var(P_k\|β) / λ_k | Per-PC **susceptibility** in dimensionless units. Peak ≈ β_c. | row 4. |
| Skew, excess kurtosis of P1 | Departure from the Gaussian assumption that KMeans/GMM make. Strong values flag heavy-tailed / asymmetric distributions. | stdout summary. |
| (P1, P2) scatter coloured by β | Visual check of cluster geometry — convex blobs, elongated arcs, or noise tail. | scatter page (RdBu, white at β_c). |
| PC1 loading on lattice / BZ | Physical interpretation of PC1. For our data, peaks at q=(0,0) (s-wave SC) and is uniform in real space (Fourier dual). | loadings page. |
| P1 ridgeline (KDE per β) | Shape evolution of P1's distribution across β: narrow → wide; symmetric → skewed. | ridgeline page. |

---

## Why we divide by σ_k or λ_k

PC scores `P_k` carry *units* — those units are the eigenvalues `λ_k = Var(P_k)`
(PCA orthogonally diagonalises the covariance matrix; eigenvalues *are* the
component variances).  PC1 and PC2 generally have very different λ's, so any
plot that puts them on the same axis without rescaling is comparing apples
to oranges.  The two natural fair comparisons are:

- **`⟨P_k⟩_β / σ_k`** with `σ_k = √λ_k` (std).  This is the per-β mean of
  PC_k expressed in *number of global standard deviations*.  Dimensionless,
  same scale for every PC.  Crossings of two PCs on this plot are
  meaningful "equal signed signal in std units".
- **`Var(P_k|β) / λ_k`**.  Dividing the *within-β variance* by the global
  variance gives a dimensionless ratio.  By construction the β-weighted
  average of this ratio is 1, so the horizontal line `y = 1` is the global
  reference.  Values above 1 mean the PC has wider-than-typical spread at
  that β (susceptibility peak); below 1, narrower.

Without these normalisations, PC1's huge variance simply dominates every
plot and PC2 looks flat by accident, not by physics.

## Why z-scoring is the right normaliser **for UMAP**, not for PCA

`StandardScaler` z-scores per axis: `x' = (x − μ) / σ`.  Each axis ends up
with std 1.  KMeans's Euclidean distance contributes
`Σ_j (x_ij − c_kj)^2`; axis `j` weighs ~ `V_j` in the sum.  Z-scoring
equalises that weighting.

UMAP's embedding has comparable, bounded, roughly-Gaussian per-axis
distributions — μ and σ are honest summaries of "where the data sits".
Z-scoring there is a clean balance correction.

PCA on heavy-tailed data inherits the heavy tails linearly.  μ and σ are
no longer summaries of the bulk — σ is inflated by the tail.  Z-scoring
PC2 in particular blows its outlier tail up to dominate the plane, and
KMeans grabs *those outliers* as a cluster.  So z-scoring is the wrong
operation for raw PCA scores.  That option lives in `umap_method.py` only.

---

## When to expect a crossing in the means overlay (row 3)

For PCA on centered data, the *β-weighted average* of `⟨P_k⟩_β` is exactly
zero (the global mean is subtracted before fitting).  So `⟨P_k⟩_β / σ_k`
**must** cross zero at some β.  The crossing of *two PCs* is more
informative than the crossing of one PC with zero:

- If they cross at low β with both ≈ 0: that's just "no signal yet, both
  PCs near origin" — not a phase boundary.
- A crossing **near β_c** with both PCs at order-1 magnitudes means PC2
  has a real subleading transition contribution — worth investigating.
- For our integrated-pair data, PC2 is essentially noise (eigenvalue gap
  λ1/λ2 ≈ 100×, susceptibility flat), so the crossing is uninformative.
  (For comparison, the **static** pair correlators measured 2026-05-02
  give λ1/λ2 ≈ 30 — still strongly 1-D but less extreme than the integrated
  case.  Density correlators sit at λ1/λ2 ≈ 1.5–1.8, so PC2 there is not
  noise.  See [[06_Experiments#Hubbard pilot — 8-observable PCA + UMAP sweep]].)

---

## Choosing the post-PCA clustering method

The four recipes we tested map onto a simple decision tree:

```
Are PC1, PC2 distributions roughly Gaussian-shaped?
  yes → "raw_2d"        (default; canonical KMeans on (P1, P2))
  no  → is PC1 heavy-tailed but its sign/direction stable across the transition?
          yes → "spherical" (drop magnitude, cluster by direction)
          no  → does PC1 grow ~exponentially with β?
                  yes → "arcsinh" (linearise the ramp; 1-D KMeans)
                  no  → consider GMM with full covariance, or move to UMAP entirely
```

Diagnostic to drive the choice (read off `pca_diagnostics`):

- **Eigenvalue spectrum** flat → no single PC carries the signal; clustering
  on PC1 alone is wrong.
- **λ1/λ2 ≫ 1 and skew(P1) > ~3** → tail problem; baseline KMeans places
  its boundary in the tail.  Use `spherical` or `arcsinh`.
- **PC1 loading is the order parameter (e.g. q=(0,0) for s-wave SC)** →
  the order parameter's *direction* is well-defined, supports the
  spherical recipe.

For the integrated-pair correlator on `L=8`, the empirical comparison
(see `plot_pca_clustering_comparison.py`):

| method | β_sig | β_deriv | gap to FSS β_c=8.5 |
|---|---|---|---|
| `raw_2d` (baseline) | 13.0 (boundary clamp — fit failed) | 11.25 | 4.50 |
| GMM full covariance | 13.0 (boundary clamp — fit failed) | 5.75 / 12.75 | 4.50 |
| `spherical` | 10.2 ± 0.11 | **8.25** | **0.25** |
| `arcsinh` | 10.2–10.5 ± 0.11 | 8.25 / 10.25 | 0.25–1.75 |

`spherical` is the most stable across both observables.  `arcsinh` works
when PC1 carries the order parameter cleanly (momentum case) but is
sensitive to which axis the order parameter sits on (position case).

---

## Choosing whether to z-score the UMAP embedding

`scale_kmeans=True` (default) z-scores the 2-D embedding before KMeans.
Use it whenever the two UMAP coordinates have noticeably different std.
Empirically:

| observable | std ratio (emb_x / emb_y) | with scaling | without |
|---|---|---|---|
| `integrated_pair_position` | 0.98 | indistinguishable from no-scaling | same |
| `integrated_pair_momentum` | 0.74 | monotonic `f_KVBS`, β_c ≈ 8.5 | non-monotonic at high β (0 → 0.84 → 0.61) |

UMAP coords are bounded and roughly Gaussian, so z-scoring is harmless
when not needed and corrective when needed.  Default `True` is safe.

---

## "Pinning" the sign of a PC loading (page 3 of diagnostics)

PCA returns a *direction* (axis), not a vector.  Both `v` and `−v` describe
the same line and the same maximised variance; the choice between them is
implementation-dependent and can flip across solver / numpy / data versions.
That flips the loading-heatmap colormap and the sign of every PC score.

To stabilise the visual we *pin* the sign deterministically: find the entry
with the largest `|·|`, and if that entry is negative, multiply the **whole
vector** by `−1` (not just one entry — flipping a single element corrupts
the loading pattern).  After pinning, `v[argmax(|v|)] ≥ 0` always.  Largest
magnitude is the most stable pivot because smaller entries flicker easily
under perturbation.

---

## Quick reference — pipeline knobs

```python
from ssh_ml_phase.pipeline import run

run(
    model              = "attractive_hubbard",
    observable         = "integrated_pair_position",
    methods            = ["pca", "umap"],
    pca_cluster_method = "spherical",   # "raw_2d" | "spherical" | "arcsinh"
    umap_scale_kmeans  = True,          # default; set False to compare
    plot               = True,
)
```

Or edit the USER CONFIG block in `run_scripts/run_*.py` and run
`python run_scripts/run_all.py`.

---

## Status of canonical results

- Canonical `pca_results_<obs>.npz` and `umap_results_<obs>.npz` use the
  defaults: PCA `cluster_method="raw_2d"`, UMAP `scale_kmeans=True`.
- Suffixed variants (`_scaled`, `_kmeans_pc1`, `_arcsinh`) from the
  exploration phase remain on disk for comparison; safe to delete once
  the chosen method is locked in.
  - **The `_scaled` UMAP variants are confirmed redundant** as of
    2026-05-02: default-tagged `umap_results_<obs>.npz` files re-run from
    that date onward are z-scaled (`scale_kmeans=True` is the default).
    The pre-default-flip Apr 30 npz files were the reason `_scaled`
    companions existed; that workaround is no longer needed.
- For the integrated-pair observables specifically, the recommendation
  going forward is `pca_cluster_method="spherical"` (closest to FSS β_c
  on both observables and most robust to the heavy PC1 tail).
  - 2026-05-02 default-`raw_2d` runs on all 8 Hubbard observables clamped
    PCA km_sig at 13.0 across the board (heavy-tailed PC1 trips the
    sigmoid fit at the upper bound), which empirically reinforces the
    `spherical` recommendation.  UMAP `raw_2d + scale_kmeans=True`,
    by contrast, lands km_sig within 0.5 of FSS on integrated observables.
    See [[06_Experiments#Hubbard pilot — 8-observable PCA + UMAP sweep]].
