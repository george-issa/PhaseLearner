# 07 — Failures and Negative Results

What did NOT work, and why we recorded it. Linked from [[00_Index]].

> Rule: do not silently delete a failed approach. Move the relevant claim or experiment here with a date and one-line cause. Future runs that reach the same dead end then cost zero time.

---

### PCA Var(P₁) is monotone on G(k)
- **Where**: `pca_results_greens0_momentum.npz`.
- **Symptom**: `argmax_β Var(P₁)(β)` lands on the right edge of the β grid (β=23) for every α — 23.0 / 22.0 / 22.0 / 23.0.
- **Root cause**: G(k) is a **monotone** order parameter (variance grows continuously with β; no peak), so the susceptibility-peak estimator is invalid here.
- **Resolution**: switch primary estimator to ⟨|P₁|⟩-derivative on G(k); use Var(P₁) only on bond-bond observables. UMAP sigmoid is the recommended primary on G(k).

### Autoencoder KMeans always lands at β≈12
- **Symptom**: KMeans (k=2) on the autoencoder's 2D latent space puts the cluster boundary at β ≈ 12 regardless of α (FSS β_c ranges 12.3–17.5).
- **Root cause**: The autoencoder picks up the **bond-bond V-shape artifact at β ≈ 12** as the most prominent latent-space feature.
- **Resolution**: never use latent KMeans as primary AE estimator. Use **Var(z₀) susceptibility peak** or reconstruction-error derivative. See [[08_Decisions_and_Pivots#Use Var(P₁) susceptibility, not KMeans, on autoencoder latents]].

### BondBond0 single correlator collapses β_c
- **Where**: `pca_results_BondBond0.npz`, `umap_results_BondBond0.npz`, plus the `_position` variants.
- **Symptom**: PCA β_c ≈ 7–10 for all α; UMAP β_c ≈ 8–10. Every estimate ~5–10 units below FSS.
- **Root cause**: A single bond-correlator label gives too few features (12 vs. 144 for the BondBond012345 stack); local fluctuations dominate over collective ordering.
- **Resolution**: abandoned. Always use BondBond012345 (all 6 labels). See [[08_Decisions_and_Pivots#Drop BondBond0 single-correlator analysis]].

### Old per-update phonon write produced 5.76 M-row CSVs
- **Symptom**: With `N_bins=100`, `bin_size=100`, the previous Julia driver wrote phonon data inside the inner HMC update loop → 10,000 measurements × 576 rows × 1 column each per simulation folder.
- **Root cause**: phonon CSV write was inside `for n in 1:bin_size`. Format was not directly loadable; downstream `extract_phonons.py` reshape was incorrect.
- **Resolution**: write moved to *after* the inner loop (once per bin) — see [[08_Decisions_and_Pivots#Phonon extraction moved out of HMC inner loop]] and `SIMULATION_UPDATES.txt` §1.

### Wrong Julia → numpy reshape (mode-major vs site-major)
- **Symptom**: Old `extract_phonons.py` did `phonon_data.reshape((N_snapshots, L*L, nph))`. Phonon spatial maps came out scrambled.
- **Root cause**: Julia stores **mode-major** (all Xa, then all Ya, then Xb, Yb), and 2D site reshape needs Fortran (column-major) order to match Julia's `n1`-fastest convention.
- **Resolution**: correct reshape is `row[:-1].reshape(4, L*L)`, then for spatial visualization `modes[i].reshape(L, L, order='F')`. Codified in `extract_phonons_edited.py` and noted in `SIMULATION_UPDATES.txt` §3.

### Old extractor ignored walkers pID=1..3
- **Symptom**: With 4 MPI walkers active, only 25% of phonon samples reached the ML-ready CSV.
- **Root cause**: `load_phonon_bin` opened only `pID=0`.
- **Resolution**: new `load_phonon_bin(folder, file_type, n_walkers=4)` reads `pID=0..n_walkers-1` and stacks row-wise.

### Attractive-Hubbard stats CSV gives only mean ± std
- **Symptom**: trying to feed `equal-time/pair/pair_momentum_equal-time_stats.csv` into the ML pipeline gave one row per β, not per-bin samples.
- **Root cause**: SmoQyDQMC's `*_stats.csv` is an aggregated mean ± std summary; per-bin samples live in HDF5.
- **Resolution**: `create_csv_pair_betasweep.py` reads HDF5 directly: per-walker `bins/bins_pID-{0..n_walkers-1}.h5`, dataset `CORRELATIONS/STANDARD/EQUAL-TIME/pair/MOMENTUM`, shape `(n_id_pairs, L, L, n_bins)` complex128. See [[02_Definitions#Attractive Hubbard pilot terms]].

### Attractive-Hubbard sign / kwarg gotchas
- `FermionPathIntegral` needs `forced_complex_potential = (U < 0)` — without it the simulation silently crashed/diverged for U=−4.
- `mt.init_mutunerlogger(... complex_sign_problem = (U < 0))` — kwarg controls element type of the *logger*, not a physical sign-problem flag; setting it wrong gives a type-mismatch error at runtime.
- `compute_correlation_ratio` uses **0-indexed** BZ; Γ = (0, 0).

### Old SLURM script bugs
- Duplicate `--partition` line ran job-name parsing through the wrong SBATCH option.
- `source_file` pointed to the previous-generation Julia driver.
- Both fixed in `run_SSH_edited2.py` along with the move to `--ntasks=4 --cpus-per-task=1` for MPI walkers.

### Old `convert_jld2_csv.jl` only processed α=0.255
- **Symptom**: only one alpha's CSVs ever showed up in `data/`.
- **Root cause**: hardcoded `a = 0.2550` instead of looping over `ALPHAS`.
- **Resolution**: now loops over `[0.235, 0.240, 0.2415, 0.2435, 0.245, 0.250, 0.255]`. Bonus fixes: duplicate `PATH_DIR`, missing `folders` re-assignment for greens momentum.
