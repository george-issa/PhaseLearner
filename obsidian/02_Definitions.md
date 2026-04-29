# 02 — Definitions

Glossary of the physics, ML, and pipeline terms that the rest of the vault treats as known. Linked from [[00_Index]].

## Physics

### α (alpha)
Electron-phonon coupling constant in the optical SSH Hamiltonian (oSSH). Sweep values: 0.240, 0.245, 0.250, 0.255 (core), plus 0.235 / 0.2415 / 0.2435 (planned). In the attractive-Hubbard pipeline preset the same `ALPHAS` slot holds `|U|` instead — see [[02_Definitions#Attractive Hubbard pilot terms]].

### β (beta) and β_c
Inverse temperature `β = 1/T` (units of `1/t`). β_c is the critical inverse temperature at which the VBS phase transition occurs. FSS reference values are in [[03_Source_Map#FSS reference β_c (SSH oSSH, w=0.5)]].

### VBS / KVBS phase
Valence Bond Solid; "K" = Kekulé, the bond pattern of the ordered phase on the honeycomb lattice. KVBS has 3 equivalent domains related by 120° rotations — the reason raw (x, y) phonon components produce a PCA *circle* rather than two clusters; see [[08_Decisions_and_Pivots#Add radial-displacement observable]].

### Bond-bond V-shape artifact
For all α tested, bond-bond correlator data shows a spurious V-shape crossover at β ≈ 12 that is **not** the phase transition. Any estimator that takes the most prominent feature in bond-bond data lands at β ≈ 12; an autoencoder's KMeans is most affected — see [[07_Failures_and_Negative_Results#Autoencoder KMeans always lands at β≈12]] and [[04_Claims_and_Proof_Obligations#PO-4 V-shape at β≈12 is observable artifact, not transition]].

### Order parameters and observables
- **BondBond012345 momentum** (default observable, 144 features): bond-bond correlator in momentum space, summed over six bond labels.
- **BondBond0**: only the first bond label; abandoned ([[08_Decisions_and_Pivots#Drop BondBond0 single-correlator analysis]]).
- **G(k) = greens0_momentum** (144 features): equal-time Green's function in momentum space; monotone order parameter — needs susceptibility-derivative or UMAP-cluster estimators, not Var(P₁) peak.
- **greens0_position**: real-space G(r); not yet tested with the current pipeline.
- **phonon_tau0** (576 features = 4 modes × 144 sites): per-bin τ=0 phonon field snapshot (Xa, Ya, Xb, Yb). 4 modes × L²; produces a circle in PCA because it carries the KVBS *direction*, not just the magnitude.
- **radial_disp** (288 features = 2 sublattices × 144 sites): per-bin time-averaged `r[i] = mean_τ √(X[i,τ]² + Y[i,τ]²)` per sublattice. Domain-invariant — preferred over `phonon_tau0` for PCA Var(P₁).

## ML estimators

### PCA susceptibility
Variance of the first principal component as a function of β: `Var(P₁)(β)`. Analogous to magnetic susceptibility — peaks at β_c when the order parameter is non-monotone. **Fails on monotone observables like G(k)**.

### Mean |P₁| derivative
Inflection-point estimator: `argmax_β |d⟨|P₁|⟩/dβ|`. Used as the secondary estimator on monotone observables (G(k), phonons). Sensitive to low-β noise — needs per-α β_min floors.

### UMAP sigmoid f_KVBS
PCA pre-reduce to 50 dims → UMAP to 2D → KMeans (k=2) → fit `f_KVBS(β) = sigmoid((β − β_c)/w)`. The β_c parameter of the sigmoid is the estimate. GMM is used as a secondary estimator.

### Assaad / Greitemann–Iqbal CNN method
Sweep over candidate β_c values; for each, train a binary CNN classifier to discriminate configurations on either side; the candidate that maximizes test accuracy is the estimate. Needs O(1000) samples per β — currently skipped pending 1000-bin DQMC data.

### Autoencoder Var(z₀) susceptibility
MLP encoder `n_features → 256 → 64 → latent_dim=2`. **Primary estimator must be Var(z₀) susceptibility peak, NOT KMeans** — KMeans on the latent space is biased to β ≈ 12 (the artifact) regardless of α; see [[08_Decisions_and_Pivots#Use Var(P₁) susceptibility, not KMeans, on autoencoder latents]].

## Pipeline

### Walker / bin / stride
- **Walker**: independent MPI Markov chain. Each gets a unique `pID` (0…N_WALKERS−1) and a different RNG seed; all share one simulation folder.
- **Bin**: one DQMC bin = `bin_size` (=100) sweeps. Per (α, β) we collect 4 walkers × 250 bins = **1000 samples**.
- **Stride**: CLI flag `--stride k` keeps every k-th row of each CSV; lets us emulate lower sample density on 1000-bin data without re-running DQMC. `--stride 10 ≈ 100 effective samples / β` (current baseline equivalent).

### File template
Pattern with `{alpha:.4f}` placeholder controlling which observable CSV the pipeline loads. Examples in [[05_Code_Map#config.py]]. Auto-tag derived from the template name is appended to result `.npz` filenames; per-α β_min lives in `BETA_MIN_PER_ALPHA` and does *not* appear in the auto-tag (use `--tag` to disambiguate).

### conda env (always activate)
**`conda activate phaselearner`** before any Python work in this repo. Used everywhere — module imports (numpy, torch, umap-learn) and notebook kernels rely on this environment.

## Attractive Hubbard pilot terms

- **U/t = −4** (single value); **⟨n⟩ = 0.875** (off half-filling, lifts SO(3) degeneracy).
- Reference T_c ≈ 0.118 t → β_c ≈ 8.5 (Paiva, dos Santos, Scalettar, Denteneer, PRB 2004) — see [[03_Source_Map#Attractive Hubbard T_c reference]].
- L = 8 pilot, then 10, 12 for FSS.
- Primary observable: on-site s-wave pair susceptibility `P_s(q)` at q = (0, 0).
- Filename: `pair_momentum_L8_n0.875_U{alpha:.4f}-1.csv` (the `{alpha:.4f}` slot holds `|U|`).
- HDF5 layout (per walker): `bins/bins_pID-{0..n_walkers-1}.h5`, dataset `CORRELATIONS/STANDARD/EQUAL-TIME/pair/MOMENTUM`, shape `(n_id_pairs, L, L, n_bins)` complex128. The stats CSV holds only mean ± std, **not per-bin samples** — extractor reads HDF5 directly. See [[07_Failures_and_Negative_Results#Attractive-Hubbard stats CSV gives only mean ± std]].
- Attractive-U gotchas:
  - `FermionPathIntegral` requires `forced_complex_potential = (U < 0)`.
  - `mt.init_mutunerlogger(... complex_sign_problem = (U < 0))` — controls the logger's element type (not physical sign-problem flag).
  - `compute_correlation_ratio` uses 0-indexed BZ, so Γ = (0, 0).
