# 08 — Decisions and Pivots

Architectural and scientific decisions that closed off branches. New decisions append below; do not edit history. Linked from [[00_Index]].

---

## Estimator decisions

### Use Var(P₁) susceptibility, not KMeans, on autoencoder latents
- **Decision**: AE primary β_c estimator is **Var(z₀) susceptibility peak**. Latent KMeans is *not* used as primary.
- **Why**: KMeans on the AE latent space is biased to β ≈ 12 (the bond-bond V-shape artifact) regardless of α — see [[07_Failures_and_Negative_Results#Autoencoder KMeans always lands at β≈12]].
- **Closes**: any future "let's just KMeans the AE" instinct.

### Drop BondBond0 single-correlator analysis
- **Decision**: do NOT run PCA / UMAP on `BondBond0_*` CSVs.
- **Why**: PCA and UMAP β_c land 5–10 units below FSS for all α; single correlator gives too few features (12 vs. 144). See [[06_Experiments#BondBond0 single correlator]].
- **Result files**: `pca_results_BondBond0*.npz` / `umap_results_BondBond0*.npz` are kept for the record but excluded from summary tables.

### UMAP on G(k) is current best β_c estimator
- **Decision**: **UMAP on `greens0_momentum`** is the headline estimator until phonon observables arrive.
- **Why**: closest to FSS across α (14.7 / 13.4 / 11.7 / 10.9 vs. FSS 17.5 / 14.3 / 12.5 / 12.3) — systematic 1–3 unit underestimate, no broken α value. PCA Var(P₁) is broken on monotone G(k); UMAP is monotone-tolerant.
- **Caveat**: replace once radial-displacement phonon data lands and beats it.

### Adopt per-α β_min floors instead of a global floor
- **Decision**: `BETA_MIN_PER_ALPHA` (per-α dict in `config.py`) is the canonical knob; global `BETA_MIN` is a fallback.
- **Why**: low-β contamination has different scales for different α (low-β disordered configs at α=0.240 sit at higher β than at α=0.255). A single global floor either under- or over-truncates somewhere. Suggested floors:
  - G(k):     `{0.240: 11, 0.245: 8, 0.250: 5, 0.255: 4}`
  - BondBond: `{0.240: 10, 0.245: 8, 0.250: 5, 0.255: 5}`
- **Tag impact**: `BETA_MIN_PER_ALPHA` does not enter the auto-tag (config-driven, not CLI). Use `--tag` to disambiguate floored vs. unfloored runs.

### Different observables get different estimators
- **Decision**: do NOT apply one-size-fits-all analysis. Match estimator to observable:
  - **BondBond012345**: Var(P₁) peak (PCA susceptibility) primary; UMAP sigmoid secondary.
  - **G(k) momentum**: UMAP sigmoid primary; ⟨|P₁|⟩-derivative secondary.
  - **G(r) position**: try UMAP first.
  - **Phonon (radial)**: UMAP + ⟨|P₁|⟩-derivative.
  - **Phonon (τ=0 raw)**: UMAP only — PCA gives a circle (3-domain KVBS).
- **Why**: Var(P₁) is monotone for monotone observables; KMeans-on-circle is meaningless. See [[07_Failures_and_Negative_Results#PCA Var(P₁) is monotone on G(k)]] and [[02_Definitions#VBS / KVBS phase]].

---

## Simulation decisions

### MPI walkers (4 × 250) instead of single 1000-bin chain
- **Decision**: use 4 MPI walkers × 250 bins/walker = 1000 samples per (α, β) point.
- **Why**: same total samples but ~4× faster wall-clock (each chain is independent and parallelizable). On the cluster this is the difference between 12-h jobs and 3-h jobs. SmoQyDQMC's `process_measurements()` averages across walkers automatically.
- **Trade-off**: 4 chains have 4× the burn-in cost. We accept it because the wall-clock saving matters more than the burn-in overhead at our current chain length.

### Phonon extraction moved out of HMC inner loop
- **Decision**: write phonon CSVs **once per bin** (after `for n in 1:bin_size` closes), not once per HMC update.
- **Why**: per-update writes generated 5.76 M-row CSVs that were not directly ML-loadable. Per-bin gives 250 rows/walker × 4 walkers = 1000 rows/(α, β) — exactly the target sample count. See `SIMULATION_UPDATES.txt` §1.

### Add radial-displacement observable
- **Decision**: emit `radial_disp_*.csv` (288 features, 2 sublattices × L²) alongside `phonon_tau0_*.csv` (576 features).
- **Why**: KVBS has 3 equivalent domains (120° rotations). Raw (x, y) phonon components encode magnitude **and** direction → PCA gives a *circle* of radius |ρ|. Collapsing to radial magnitude `r[i] = mean_τ √(X[i,τ]² + Y[i,τ]²)` is domain-invariant → ordered phase separates as two blobs.
- **Trade-off**: lose access to per-domain analysis, but for a β_c estimator we only need the magnitude.

### Run `convert_jld2_csv` over all α in one Julia invocation
- **Decision**: `convert_jld2_csv_edited.jl` loops over `ALPHAS` instead of taking one hardcoded value.
- **Why**: the old hardcoded `a = 0.2550` silently produced output for one α; we lost time before noticing the missing files. See [[07_Failures_and_Negative_Results#Old `convert_jld2_csv.jl` only processed α=0.255]].

---

## Pipeline decisions

### Auto-tag from `--file-template` instead of mandatory `--tag`
- **Decision**: `make_tag()` derives the result-file suffix from `--file-template` automatically; `--tag` becomes optional.
- **Why**: every observable used to need an explicit `--tag`, easy to forget → results overwriting each other. Auto-tag prevents accidental clobber.
- **Caveat**: per-α floors don't enter the auto-tag — for a floored vs. unfloored comparison, still pass `--tag` manually.

### Stride flag for sample-density studies
- **Decision**: `--stride k` keeps every k-th row; lets us emulate 100-, 200-, 500-, 1000-sample density on the same 1000-bin data.
- **Why**: with 1000-bin data we can answer "how few samples are enough?" without re-running DQMC. Future planning aid.

### Attractive-Hubbard pilot reads HDF5 directly, not stats CSV
- **Decision**: `create_csv_pair_betasweep.py` opens `bins/bins_pID-{0..n_walkers-1}.h5` and pulls `CORRELATIONS/STANDARD/EQUAL-TIME/pair/MOMENTUM`.
- **Why**: SmoQyDQMC's `*_stats.csv` only ships mean ± std (one row per β). The ML pipeline needs **per-bin samples**. See [[07_Failures_and_Negative_Results#Attractive-Hubbard stats CSV gives only mean ± std]].

### Model registry (`MODEL_CONFIGS`) instead of branch-per-model
- **Decision**: keep one codebase; switch model via `apply_model(name)` which rebinds globals (ALPHAS, FSS, FILE_TEMPLATE, L, β bounds). `--model {ssh,holstein,bond_holstein,attractive_hubbard}` on the CLI.
- **Why**: avoid forking `ssh_ml_phase` per Hamiltonian. Lazy imports in `run_all.py` pick up the patched globals automatically.

---

## Stage / architecture revisions

### Stage A — single observable, single estimator
- BondBond012345 only; PCA susceptibility only; ~100 samples / β.
- Result: works for α=0.240 (17.0 vs. 17.5 FSS); degrades for higher α.

### Stage B — multi-observable, per-method estimators
- Added G(k) (greens0_momentum); discovered Var(P₁) is monotone there; introduced UMAP sigmoid + ⟨|P₁|⟩-derivative.
- Result: UMAP on G(k) becomes the new headline estimator.

### Stage C — per-α β_min, multi-walker DQMC, model registry
- Per-α β_min floors in `config.py`.
- 4-walker × 250-bin DQMC on 277 jobs (target: 1000 samples / β).
- Phonon observables (`phonon_tau0`, `radial_disp`) added to the extractor.
- Model registry (`MODEL_CONFIGS`) with `--model` CLI flag.
- `make_tag()` auto-tag.
- Result: same pipeline now serves SSH, Holstein, bond-Holstein, and attractive-Hubbard.

### Stage D (in flight) — second-model port to attractive Hubbard
- HDF5-direct extractor, attractive-U gotchas codified, smoke test passes.
- Pending: full β-sweep + ML pipeline run; closes [[04_Claims_and_Proof_Obligations#PO-5 Hubbard pilot recovers β_c ≈ 8.5 with current pipeline]].
- After Stage D: extend Hubbard L=8 → L=10, 12 for finite-size scaling.
