# 06 — Experiments

Append-only ledger of runs. Each entry: date, command, dataset, observable, output, takeaway. Linked from [[00_Index]] and [[01_Project_State]].

## SSH oSSH runs (~100 samples / β baseline)

### Baseline PCA — BondBond012345 momentum
- **Date**: pre-2026-02-23 (from `RECOMMENDATIONS.txt`).
- **Command**: `python -m ssh_ml_phase --skip-ae --skip-assaad --skip-umap` with default `FILE_TEMPLATE`.
- **Output**: `results/pca_results_BondBond012345_momentum.npz`.
- **β_c estimates**: 17.0 / 15.5 / 14.0 / 19.0 (α=0.240/.245/.250/.255) vs. FSS 17.5 / 14.3 / 12.5 / 12.3.
- **Takeaway**: Best PCA estimator. Clean for α=0.240; degrades and finally fails at α=0.255 (Var(P₁) becomes monotone).

### Baseline UMAP — BondBond012345 momentum
- **Output**: `results/umap_results_BondBond012345_momentum.npz`.
- **β_c**: 12.5 / 13.6 / 12.6 / 13.3 — dragged toward the V-shape artifact at β ≈ 12.

### bmin8 floor sweep
- **Command**: `python -m ssh_ml_phase --beta-min 8 --skip-ae --skip-assaad`.
- **Output**: `pca_results_bmin8.npz`, `umap_results_bmin8.npz`.
- **β_c (PCA)**: 17.0 / 15.5 / 14.0 / 19.0 — floor-insensitive at b=8 (the V-shape contamination is below β=8 already).
- **β_c (UMAP)**: 15.7 / 15.2 / 14.3 / 15.5 — clear improvement vs. unfloored UMAP. Per-α floors should improve further; see [[04_Claims_and_Proof_Obligations#PO-1 Per-α β_min floors close UMAP G(k) gap]].

### PCA / UMAP on G(k) (greens0_momentum)
- **Output**: `pca_results_greens0_momentum.npz`, `umap_results_greens0.npz`.
- **PCA Var(P₁) β_c**: 23.0 / 22.0 / 22.0 / 23.0 — broken (monotone variance, argmax pinned to grid edge β=23). See [[07_Failures_and_Negative_Results#PCA Var(P₁) is monotone on G(k)]].
- **PCA ⟨|P₁|⟩-derivative β_c**: 21.8 / 19.2 / 15.2 / 18.8 — needs per-α β_min floors.
- **UMAP β_c**: 14.7 / 13.4 / 11.7 / 10.9 — **best overall** estimator before per-α floors.

### BondBond0 single correlator
- **Output**: `pca_results_BondBond0.npz`, `pca_results_BondBond0_position.npz`, `umap_results_BondBond0.npz`, `umap_results_BondBond0_position.npz`.
- **β_c (PCA)**: 7.0 / 10.0 / 8.5 / 9.5 — collapsed.
- **Takeaway**: Single correlator is starved of features. Decision: [[08_Decisions_and_Pivots#Drop BondBond0 single-correlator analysis]].

### electron_densities w1.00 (Holstein quick check)
- **Output**: `results/pca_results_electron_densities_w1.00.npz`, `results/umap_results_electron_densities_w1.00.npz`.
- **Status**: smoke test for the Holstein preset; numerics not folded into the SSH summary.

## SSH oSSH runs (1000 samples / β — pending)

### 1000-bin DQMC sweep
- **Status**: 277 SLURM jobs queued (4 walkers × 250 bins / point). See [[03_Source_Map#Per-α β grids (SSH oSSH)]].
- **Once data lands**: re-run PCA + UMAP, then enable Assaad ([[04_Claims_and_Proof_Obligations#PO-3 Assaad CNN converges with ≥1000 samples / β]]).
- **Stride study planned**: `for s in 10 5 2 1` to learn the minimum sample density that gives stable β_c — informs simulation budget.

### Phonon observables (pending DQMC code re-run)
- **Files expected**:
  - `data/phonon_tau0_L12_w0.5_m0.0_a{α}-1.csv` (1000 × 577 cols)
  - `data/radial_disp_L12_w0.5_m0.0_a{α}-1.csv` (1000 × 289 cols)
- **Plan**: visualize first (`python -m ssh_ml_phase.visualize`), then UMAP + ⟨|P₁|⟩-derivative on radial-disp. Closes [[04_Claims_and_Proof_Obligations#PO-2 Phonon radial displacement gives a clean monotone signal]].

## Hubbard pilot (2D attractive square)

### Hubbard pilot smoke test (β=8)
- **Date**: 2026-04-28.
- **Parameters**: U/t = −4, ⟨n⟩ = 0.875, L = 8, β = 8.0, 4 walkers × 100 bins.
- **Result**: P_s(q=0) ≈ 5.36 ± 0.37; off-Γ values ≈ 0.5–0.7 → ~10× SC enhancement.
- **Takeaway**: Simulation produces the expected SC signal; β_c ≈ 8 is bracketed. Pilot β-sweep (β ∈ [2, 14] step 0.5) is in `sims/` and queued for full submission. Reference T_c ≈ 0.118 t → β_c ≈ 8.5.

### Hubbard pilot β-sweep (queued)
- **Status**: per-β `.jl` and `.log` files visible in `sims/hubbard_square/hubbard_square_U4.0000_n0.875_L8_b{β}.00-1.{jl,log}` for β ∈ [2.0, 14.0]. Some `.log` files are present (smoke runs); full-1000-bin sweep queued.
- **Once complete**: extract via `create_csv_pair_betasweep.py`, then edit `run_scripts/run_all.py` (`MODEL = "attractive_hubbard"`, `OBSERVABLE = "..."`) and run `python run_scripts/run_all.py`. Closes [[04_Claims_and_Proof_Obligations#PO-5 Hubbard pilot recovers β_c ≈ 8.5 with current pipeline (closed 2026-05-02)]].

### Hubbard pilot — 8-observable PCA + UMAP sweep
- **Date**: 2026-05-02.
- **Parameters**: U = −4, ⟨n⟩ = 0.875, L = 8, β ∈ [2, 14] step 0.5, ~400 samples/β (4 walkers × 100 bins).
- **Observables**: 4 static (`{density,pair} × {position,momentum}`) + 4 integrated susceptibility counterparts (`integrated_*`).
- **Output**: `results/npz/{pca,umap}_results_<obs>.npz` (8 + 8 files); per-observable PDFs at `results/figures/attractive_hubbard/{pca,umap}/<obs>/{avg_analysis,embedding,PC_layout}.pdf` (PCA) and `embedding.pdf` (UMAP).

**β_c estimates** (FSS reference β_c = 8.5):

| observable                    | PCA km_sig    | PCA sm-argmax_P1 | UMAP km_sig   | UMAP km_deriv |
|-------------------------------|---------------|------------------|---------------|---------------|
| density_position              | 11.0 ± 0.83   | 9.5              | 12.50 ± 1.18  | 13.75         |
| density_momentum              | 13.0 ± 0.51   | 9.5              | 11.97 ± 1.03  | 13.75         |
| pair_position                 | 13.0 ± 0.51   | 9.5              | 11.73 ± 0.56  |  5.25         |
| pair_momentum                 | 13.0 ± 0.51   | 9.5              | 13.00 ± 0.60  |  4.75         |
| integrated_density_position   | (see below)   | 9.5              |  9.09 ± 0.21  |  9.25         |
| integrated_density_momentum   | (see below)   | 9.5              |  8.67 ± 0.20  |  8.75         |
| integrated_pair_position      | (see below)   | 9.5              |  8.04 ± 0.10  |  8.25         |
| integrated_pair_momentum      | 13.0 ± 0.48   | 9.5              |  8.53 ± 0.11  |  8.25         |

PCA km_sig on integrated observables boundary-clamps at the same 13.0 across the board (heavy-tailed PC1 fools `raw_2d` KMeans on every observable here — the prior `spherical`/`arcsinh` recipe work in [[10_PCA_UMAP_Clustering]] is the right path forward for the PCA-cluster estimator).

- **Takeaway 1 — integrated ≫ static.**  UMAP km_sig on the four integrated correlators lands within ~0.5 of FSS β_c = 8.5; static correlators are off by 3–5 units.  PCA Var-based sm-argmax_P1 = 9.5 across **every** observable (same Var(P₁) peak whether the input is static or integrated) — close, but the integrated UMAP path is the tightest.
- **Takeaway 2 — PCA eigenvalue gap λ1/λ2.**  Pair correlators ≈ 30 (essentially 1-D order-parameter axis); density correlators ≈ 1.5–1.8 (PC2 still carries signal).  Pair → diagnostic-style PC1 loading is interpretable; density → expect more PC2 mixing.
- Closes [[04_Claims_and_Proof_Obligations#PO-5 Hubbard pilot recovers β_c ≈ 8.5 with current pipeline (closed 2026-05-02)]].

## Result-file checkpoints (in `results/`)

| File                                                         | Method | Observable                          | Notes |
|--------------------------------------------------------------|--------|-------------------------------------|-------|
| `pca_results.npz`                                            | PCA    | (default — pre-auto-tag run)        | older artifact, leave in place |
| `pca_results_bmin8.npz`                                      | PCA    | BondBond012345 momentum, β_min = 8  | floor experiment |
| `pca_results_BondBond0.npz`                                  | PCA    | BondBond0 momentum                  | abandoned method |
| `pca_results_BondBond0_position.npz`                         | PCA    | BondBond0 position                  | abandoned method |
| `pca_results_BondBond012345_momentum.npz`                    | PCA    | BondBond012345 momentum (default)   | best PCA |
| `pca_results_greens0.npz`                                    | PCA    | greens0 (legacy file)               | superseded by `_momentum` |
| `pca_results_greens0_momentum.npz`                           | PCA    | greens0_momentum                    | Var(P₁) broken; see ⟨|P₁|⟩-deriv |
| `pca_results_electron_densities_w1.00.npz`                   | PCA    | Holstein                            | sanity check |
| `umap_results.npz`                                           | UMAP   | (default)                           | older artifact |
| `umap_results_bmin8.npz`                                     | UMAP   | BondBond012345 + β_min = 8          | clear improvement vs. default |
| `umap_results_BondBond0.npz`                                 | UMAP   | BondBond0 momentum                  | abandoned |
| `umap_results_BondBond0_position.npz`                        | UMAP   | BondBond0 position                  | abandoned |
| `umap_results_BondBond012345_momentum.npz`                   | UMAP   | BondBond012345 momentum             | dragged to β=12 artifact |
| `umap_results_greens0.npz`                                   | UMAP   | greens0 (legacy)                    | best overall β_c |
| `umap_results_electron_densities_w1.00.npz`                  | UMAP   | Holstein                            | sanity check |
