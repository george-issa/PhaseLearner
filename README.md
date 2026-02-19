# PhaseLearner

Machine learning methods for detecting phase transitions in the optical SSH (Su-Schrieffer-Heeger) model. Four unsupervised/semi-supervised approaches are benchmarked against finite-size scaling (FSS) reference values for the valence bond solid (VBS) critical inverse temperature $\beta_c$.

## Physical Context

All models are use an $L \times L$ lattice, with $L =12$, filled with bond-bond correlation data in momentum space. The key parameters are:

- **$\alpha$** — coupling constant (0.240, 0.245, 0.250, 0.255)
- **$\beta$** — inverse temperature; the goal is to identify the critical value β_c where the VBS phase transition occurs
- **$\omega= 0.5$** — phonon frequency

All parameters are in units of $t$, the hopping amplitude of electrons appearing in the kinetic term of the Hamiltonian.

FSS reference values:

| $\alpha$  | $\beta_c$ |
| --------- | --------- |
|   0.240   |   17.5    |
|   0.245   |   14.3    |
|   0.250   |   12.5    |
|   0.255   |   12.3    |

## Methods

### 1. PCA (`pca_method.py`)
Principal Component Analysis on the $L^2$-dimensional bond-bond correlator configurations. $\beta_c$ is estimated from:
- **PCA susceptibility**
Peak of $Var(P_1)$ and $Var(P_2)$ as a function of $\beta$ (analogous to magnetic susceptibility)
- **KMeans clustering**
$k=2$ clustering in $(P_1, P_2)$ space; sigmoid fit to the KVBS fraction $f_{KVBS}(\beta)$

### 2. UMAP (`umap_method.py`)
Non-linear dimensionality reduction to a $2$-dimensional embedding, followed by:
- **KMeans clustering** 
and sigmoid fit to $f_{KVBS}(\beta)$
- **GMM (Gaussian Mixture Model)** 
clustering as a secondary estimate

### 3. Assaad / Greitemann–Iqbal method (`assaad_method.py`)
CNN-based approach: trains a classifier to distinguish configurations on either side of a guessed $\beta_c$, then sweeps over candidate values. The true $\beta_c$ maximizes test accuracy. Needs significantly more samples than the rest of the methods!

### 4. Autoencoder (`autoencoder.py`)
An autoencoder with a $2$-dimensional bottleneck (`latent_dim=2`) trained on all configurations. $\beta_c$ is estimated from:
- **Reconstruction error** 
The reconstruction error is highest at $\beta_c$ since ambiguity is notable accross many samples
- **Latent KMeans**
$k=2$ clustering in latent space $(z_0, z_1)$ with sigmoid fit

## Installation

```bash
pip install -e .
```

For UMAP support:

```bash
pip install -e ".[umap]"
```

**Requirements:** Python ≥ 3.8, numpy, scipy, scikit-learn, torch, matplotlib. Optional: umap-learn.

## Data

Place input CSV files in a `NEW/` directory at the project root. Each file should be named:

```
BondBond012345_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv
```

Each row contains $144$ momentum-space features (flattened $12\times12$ lattice) followed by the inverse temperature $\beta$.

## Usage

### Run all methods

```bash
python -m ssh_ml_phase
```

### Run individual methods

```bash
python -m ssh_ml_phase.pca_method
python -m ssh_ml_phase.umap_method
python -m ssh_ml_phase.assaad_method
python -m ssh_ml_phase.autoencoder
```

### Single alpha (fast iteration)

```bash
python -m ssh_ml_phase.pca_method --alphas 0.240
```

### Skip slow methods

```bash
python -m ssh_ml_phase --skip-assaad --skip-ae
```

Results are saved as `.npz` files in the `results/` directory.

## Visualization

Open `SSH_ML_phase_detection.ipynb` after running the methods. The notebook loads the pre-computed `.npz` result files and generates comparison plots saved to `ssh_ml_phase/figures/`:

| Figure | Content |
|--------|---------|
| `PCA_susceptibility_argmax.pdf` | $Var(P_1)$, $Var(P_2)$, and their derivatives vs $\beta$ |
| `PCA_kmeans.pdf` | $2D$ PCA embedding and KMeans $f_{KVBS}$ curves |
| `UMAP_beta_c.pdf` | UMAP embedding and KMeans/GMM $f_{KVBS}$ curves |
| `Part4_autoencoder.pdf` | Mean reconstruction error, derivative, and latent order parameter |
| `Part4_latent_scatter.pdf` | Autoencoder latent space scatter coloured by $\beta$ and cluster |
| `Final_summary_all_methods.pdf` | All methods' $\beta_c$ estimates vs FSS reference |

## Project Structure

```
PhaseLearner/
├── ssh_ml_phase/
│   ├── config.py          # FSS references, paths, physical constants
│   ├── data.py            # CSV loading utilities
│   ├── pca_method.py      # PCA susceptibility and KMeans
│   ├── umap_method.py     # UMAP + KMeans/GMM
│   ├── assaad_method.py   # Greitemann–Iqbal CNN method
│   ├── autoencoder.py     # Autoencoder anomaly detection
│   ├── run_all.py         # Orchestrates all methods
│   └── figures/           # Output plots
├── results/               # Saved .npz result files (generated)
├── NEW/                   # Input CSV data (not tracked)
├── SSH_ML_phase_detection.ipynb
└── setup.py
```
