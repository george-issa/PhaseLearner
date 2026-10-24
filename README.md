# PhaseLearner

Unsupervised and semi-supervised machine learning for locating finite-temperature
phase transitions in determinant quantum Monte Carlo (DQMC) data.

The main target is the **optical Su-Schrieffer-Heeger (oSSH) model** on the
honeycomb lattice ($L = 12$, $\omega = 0.5$), where the electron-phonon coupling
$\alpha$ drives a Kekulé valence-bond-solid (KVBS) transition. Each method
estimates the critical inverse temperature $\beta_c$ from equal-time or
$\tau$-integrated correlators, and the result is checked against finite-size
scaling (FSS):

| $\alpha$  | 0.240 | 0.245 | 0.250 | 0.255 |
| --------- | ----- | ----- | ----- | ----- |
| $\beta_c$ (FSS) | 17.5 | 14.3 | 12.5 | 12.3 |

All energies are in units of the hopping $t$. The same pipeline also runs on the
Holstein, bond-Holstein and attractive-Hubbard models (see `MODEL_CONFIGS` in
`ssh_ml_phase/config.py`).

## Methods

| Key       | Module                 | How $\beta_c$ is read off |
| --------- | ---------------------- | ------------------------- |
| `pca`     | `pca_method.py`        | Peak of the PCA susceptibility $\mathrm{Var}(P_1)$, or of $d\langle\lvert P_1\rvert\rangle/d\beta$ for monotone observables |
| `umap`    | `umap_method.py`       | PCA → UMAP embedding, KMeans/GMM clustering, sigmoid fit to the ordered fraction |
| `tsne`    | `tsne_method.py`       | PCA → t-SNE map; cluster structure and the density "neck" along the $\beta$ path |
| `ae`      | `autoencoder.py`       | Autoencoder with a 2D latent space; peak of $\mathrm{Var}(z_0)$ |
| `predict` | `prediction_method.py` | MLP regresses $\beta$; peak of $d\langle\beta_\mathrm{pred}\rangle/d\beta$ |
| `mean`    | `mean_method.py`       | Optimal-regressor limit of `predict` (Arnold et al. 2021) |
| `assaad`  | `assaad_method.py`     | Classifier sweep over trial $\beta_c$ (Greitemann–Iqbal); needs ~1000 samples per $\beta$ |
| `lbc`     | `lbc_method.py`        | Learning by confusion (van Nieuwenburg et al. 2017); W-shaped accuracy curve |

## Repository layout

```
PhaseLearner/
├── ssh_ml_phase/      Python package: config, data loading, methods, plotting, pipeline
├── scripts/           Entry points for runs, sweeps, summaries and diagnostic plots
├── sims/
│   ├── ssh/           oSSH DQMC driver (Julia), SLURM submitters, JLD2/HDF5 → CSV extraction
│   └── hubbard_square/  Attractive-Hubbard pilot: driver, runner, HDF5 → CSV extraction
├── lbc_validation/    Known-answer test of learning by confusion on the 2D Ising model
├── paper/             Paper figure scripts (paper/scripts) and note figures (paper/notes/figs)
├── notebooks/         Exploratory and plotting notebooks
├── results/           Method outputs (.npz) and summary tables
├── environment.yml    Conda environment (Python side)
├── Project.toml       Julia environment (DQMC side, SmoQyDQMC + MPI)
└── setup_local.sh     One-shot local setup of both environments
```

Input data live in `data/<model>/` and are not tracked.

## Installation

```bash
conda env create -f environment.yml    # creates "phaselearner" and installs the package
conda activate phaselearner
```

To also set up the Julia environment for running simulations locally:

```bash
bash setup_local.sh
```

## Usage

Every run goes through `scripts/run_all.py`. Edit its DATA block (model,
observables, $\alpha$ values, $\beta$ window) and METHODS block (which methods
to run and their settings), then:

```bash
python scripts/run_all.py
```

Results are written to `results/npz/<method>_results_<tag>.npz`, and figures
(with `PLOT = True`) to `results/figures/<model>/<method>/<tag>/`. Setting `SID`
nests both under `results/<SID>/`. To re-render figures from existing results
without recomputing:

```bash
python scripts/run_plot.py
```

Each input CSV holds one configuration per row: the flattened correlator
features followed by $\beta$ in the last column.

## Simulations

`sims/ssh/` holds the oSSH DQMC driver and everything needed to go from raw
output to ML-ready CSVs. Run the submitters from the repo root, for example
`python sims/ssh/run_SSH_edited2.py`; `sims/ssh/RUN_PROCEDURE.md` walks through
a full workstation run. The attractive-Hubbard pilot in `sims/hubbard_square/`
follows the same pattern.

## License

MIT; see `LICENSE`.
