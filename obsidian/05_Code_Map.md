# 05 — Code Map

File-by-file tour of the PhaseLearner codebase. Linked from [[00_Index]] and [[01_Project_State]].

> Reminder: every Python entry point assumes `conda activate phaselearner` has run.

## Repo layout

```
PhaseLearner/
├── ssh_ml_phase/                 # the Python package
│   ├── __init__.py
│   ├── __main__.py               # `python -m ssh_ml_phase` entry point
│   ├── config.py                 # FSS values, paths, FILE_TEMPLATE, model registry
│   ├── data.py                   # CSV loading, β masking, auto-tag generation
│   ├── pca_method.py             # PCA susceptibility + KMeans + ⟨|P₁|⟩-derivative
│   ├── umap_method.py            # PCA→UMAP→KMeans/GMM + sigmoid f_KVBS fit
│   ├── assaad_method.py          # CNN classifier sweep (Greitemann–Iqbal)
│   ├── autoencoder.py            # MLP autoencoder; Var(z₀) susceptibility primary
│   ├── plot.py                   # shared matplotlib helpers (tracked but new file)
│   ├── visualize.py              # --visualize: feature heatmaps + PCA scatter + phonon maps
│   └── run_all.py                # orchestrates PCA/UMAP/Assaad/AE
├── data/                          # input CSVs (BondBond, greens, phonon, electron_densities, ...)
│   └── hubbard_square/            # Hubbard pilot raw outputs (per-β subfolders)
├── results/                       # `.npz` outputs of each method
├── figures/                       # PDFs (PCA / UMAP / autoencoder / final summary)
├── sims/                          # generated SLURM .jl + .sh files for the Hubbard pilot
├── obsidian/                      # this knowledge base
├── SSH_ML_phase_detection.ipynb   # publication-figure notebook (loads .npz from results/)
├── README.md
├── RECOMMENDATIONS.txt           # 2026-02-23 analysis & next-step memo
├── SIMULATION_UPDATES.txt        # 2026-02-23 simulation/pipeline update memo
├── BondHolstein_paper.pdf        # source for FSS β_c (bond-Holstein model)
├── next_steps_feb23.{tex,pdf}    # written-up plan from late Feb 2026
├── setup.py / setup_local.sh
├── environment.yml               # conda env spec → conda activate phaselearner
├── Project.toml / Manifest.toml / LocalPreferences.toml   # Julia env (cluster mirror)
├── create_csv_bondbond_betasweep.py
├── create_csv_greens_betasweep.py
├── create_csv_pair_betasweep.py  # Hubbard pilot HDF5 → CSV
├── extract_phonons_edited.py     # local copy of cluster phonon extractor
├── run_attractive_hubbard.py     # local/SLURM Hubbard runner
├── run_SSH_array_alpha240_pending.py
├── run_SSH_edited2.py
├── scratch.sh
└── %OUTDIR%                      # leftover from a Julia/SLURM template (rename or remove on next sweep)
```

## ssh_ml_phase package

### config.py
- Holds all globals and the **model registry** `MODEL_CONFIGS` with presets `"ssh"` (default), `"holstein"`, `"bond_holstein"`, `"attractive_hubbard"`.
- `apply_model(name)` rebinds module-level globals for the chosen preset; called by `run_all.py` before lazy method imports so every method picks up the new state.
- Per-α β bounds live in `BETA_MIN_PER_ALPHA` / `BETA_MAX_PER_ALPHA` (looked up first, with global `BETA_MIN/MAX` as fallback). These do *not* enter the auto-tag — use `--tag` to disambiguate runs.
- Suggested floors live as comments in the file (G(k) and BondBond), see [[01_Project_State#Next steps — Priority A (no new DQMC)]].

### data.py
- CSV loader: every row = `[features..., β]`; loader returns `(X, y_beta)` arrays per α.
- `make_tag()` auto-generates the result tag from `--file-template` (falls back to `DEFAULT_FILE_TEMPLATE`).
- `load_alpha_sweep()` is for fixed-β files (varying α); referenced by RECOMMENDATIONS.txt for the future alpha-sweep mode.
- β masking applies global + per-α bounds.

### pca_method.py
- Two β_c estimators:
  1. `sm_argmax_P1` — smoothed argmax of Var(P₁); the susceptibility peak.
  2. `mean_P1_deriv_bc` — argmax of |d⟨|P₁|⟩/dβ|; for monotone observables (G(k), phonons).
- KMeans (k=2) in (P₁, P₂) plane → sigmoid fit to f_KVBS (secondary).
- ⚠️ Var(P₁) is monotone for G(k) — use `mean_P1_deriv_bc` there. See [[07_Failures_and_Negative_Results#PCA Var(P₁) is monotone on G(k)]].

### umap_method.py
- PCA pre-reduce to 50 dims → UMAP to 2D → KMeans (k=2) → sigmoid `f_KVBS(β)` → β_c is the sigmoid centre.
- GMM as secondary cluster method.
- Best overall β_c estimator on G(k) right now ([[01_Project_State]] table).

### assaad_method.py
- CNN sweep: at each candidate β_c, train binary classifier; take argmax of test accuracy.
- Skipped by default until 1000-bin data is in place ([[04_Claims_and_Proof_Obligations#PO-3 Assaad CNN converges with ≥1000 samples / β]]).

### autoencoder.py
- MLP encoder `n_features → 256 → 64 → latent_dim=2`.
- **Primary β_c estimator**: Var(z₀) susceptibility peak. **Do NOT use latent KMeans** — biases to β ≈ 12 artifact regardless of α ([[08_Decisions_and_Pivots#Use Var(P₁) susceptibility, not KMeans, on autoencoder latents]]).
- Reconstruction-error derivative is a backup estimator.

### visualize.py
- Diagnostics module (`python -m ssh_ml_phase.visualize`).
- Part A — feature heatmap, variance vs β, PCA(2) scatter coloured by β.
- Part B — phonon spatial maps (`phonon_tau0` quivers, `radial_disp` heatmaps) at representative β.
- Use it before launching ML to sanity-check new observables.

### run_all.py
- Wires `--model`, `--file-template`, `--alphas`, `--beta-min/max`, `--beta-min-per-alpha`, `--stride`, `--skip-{assaad,umap,ae}`, `--plot`, `--tag` flags.
- Calls `config.apply_model(name)` first, then lazy-imports each method module so they read the patched globals.

## Cluster-side scripts

(also referenced from [[03_Source_Map#Project files on cluster]])

- `run_SSH_edited2.py` — generates and submits 277 SLURM jobs (4 ntasks each → 4 walkers × 250 bins = 1000 samples / point). Per-α β grids in [[03_Source_Map#Per-α β grids (SSH oSSH)]].
- `run_SSH_array_alpha240_pending.py` — one-off submitter for the α=0.240 extension (β to ≈30).
- `extract_phonons_edited.py` — combines per-walker phonon CSVs from `Phonon_positions/` into ML-ready `phonon_tau0_*` and `radial_disp_*`. Correct reshape: `row[:-1].reshape(4, L*L)` then `modes[i].reshape(L, L, order='F')` (Julia is column-major).
- `convert_jld2_csv_edited.jl` — JLD2 → per-β CSV; loops over `ALPHAS = [0.235, 0.240, 0.2415, 0.2435, 0.245, 0.250, 0.255]` (vs. old hardcoded 0.255).
- `attractive_hubbard_square_mpi_execute.jl` — Hubbard pilot driver; uses `HubbardModel`, `HubbardSpinHirschHST`, MuTuner. Attractive-U pitfalls: `forced_complex_potential = (U < 0)`, `complex_sign_problem = (U < 0)` for the MuTuner logger, Γ at (0, 0).
- `run_attractive_hubbard.py` — `LOCAL` flag toggles between `mpirun` direct and `sbatch`; auto-detects `mpiexecjl`/`mpirun`.
- `create_csv_pair_betasweep.py` — reads HDF5 directly (`bins/bins_pID-{0..n_walkers-1}.h5`, dataset `CORRELATIONS/STANDARD/EQUAL-TIME/pair/MOMENTUM`); the stats CSV only carries mean ± std, *not* per-bin samples.

## Notebook

`SSH_ML_phase_detection.ipynb` loads `.npz` files in `results/` and produces the publication PDFs in `figures/`. Don't put numerics there — only plotting.

## Loose ends

- `%OUTDIR%` directory in the repo root is leftover from a SLURM template; rename or remove on the next clean-up sweep.
- `RECOMMENDATIONS.txt` and `SIMULATION_UPDATES.txt` are still authoritative as human-readable plan docs; this Obsidian vault is the structured mirror — **don't delete them yet**, but do prefer this vault for new prose.
