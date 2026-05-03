"""
Shared configuration: FSS reference values, paths, and physical constants.

Model presets (--model flag)
-----------------------------
Call apply_model(name) to switch between model configurations at runtime.
Available: "ssh"  (oSSH, default), "holstein", "bond_holstein".
"""
import numpy as np
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NPZ_DIR = os.path.join(RESULTS_DIR, "npz")

# ── Physical parameters ───────────────────────────────────────────────────────
L = 12
SIZE = L * L  # 144 unit cells; phonon observables have 576 (tau0) or 288 (radial) features

# All alpha values — original 4 + new additions (see RECOMMENDATIONS.txt)
# ALPHAS = [0.235, 0.240, 0.2415, 0.2435, 0.245, 0.250, 0.255]
ALPHAS = [0.240, 0.245, 0.250, 0.255]

# FSS reference values (w=0.5, from sohan_phase_diagram.pdf)
# New alphas 0.235/0.2415/0.2435: beta_c TBD from FSS once 1000-bin data arrive.
FSS_ALPHA  = np.array([0.240,  0.245,  0.250,  0.255])
FSS_BETA_C = np.array([17.5,   14.3,   12.5,   12.3])

# Mapping from alpha to FSS beta_c (NaN returned for alphas without FSS reference)
FSS_DICT = dict(zip(FSS_ALPHA, FSS_BETA_C))

# ── Model registry ─────────────────────────────────────────────────────────────
# Named presets for each physical model.  Select via --model on the CLI.
# FSS_BETA_C = NaN means no finite-size-scaling reference is available yet.
#
# Holstein coupling convention: alpha = g (bare e-ph strength).
#   Dimensionless coupling λ_D = 2g²/(ω₀ W) with ω₀=1, W=8t → λ_D = g²/4.
#   a0.6325 → λ_D≈0.10,  a0.7071 → λ_D≈0.125,  a0.8165 → λ_D≈0.167,
#   a1.0000 → λ_D=0.25 (β_c≈6),  a1.4142 → λ_D=0.50 (β_c≈2.5).
#
# Bond Holstein: phonons couple to the hopping (bond), not the site energy.
#   Observable and FSS β_c values are TBD — update FILE_TEMPLATE when
#   the data CSVs are produced.
MODEL_CONFIGS = {
    "ssh": {
        "ALPHAS":            [0.240, 0.245, 0.250, 0.255],
        "FSS_ALPHA":         np.array([0.240,  0.245,  0.250,  0.255]),
        "FSS_BETA_C":        np.array([17.5,   14.3,   12.5,   12.3]),
        "FILE_TEMPLATE":     "BondBond012345_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv",
        "DATA_SUBDIR":       "ssh/csv",   # appended to base data/ on apply_model
        "L":                 12,
        "BETA_MIN":          None,
        "BETA_MAX":          None,
        "BETA_MIN_PER_ALPHA": {},
        "BETA_MAX_PER_ALPHA": {},
    },
    "holstein": {
        "ALPHAS":            [0.6325, 0.7071, 0.8165, 1.0000, 1.4142],
        "FSS_ALPHA":         np.array([0.6325, 0.7071, 0.8165, 1.0000, 1.4142]),
        "FSS_BETA_C":        np.array([np.nan, np.nan, np.nan,  6.0,    2.5]),
        "FILE_TEMPLATE":     "electron_densities_w1.00_a{alpha:.4f}_L12-1.csv",
        "DATA_SUBDIR":       "holstein/csv",
        "L":                 12,
        "BETA_MIN":          None,
        "BETA_MAX":          None,
        "BETA_MIN_PER_ALPHA": {},
        "BETA_MAX_PER_ALPHA": {},
    },
    "attractive_hubbard": {
        # 2D square-lattice attractive Hubbard model.
        # Pipeline convention: ALPHAS holds |U| values (single-precision sweep over U).
        # First pilot: U=-4, ⟨n⟩=0.875, L=8, β∈[2,14] step 0.5.
        # Reference T_c at U=-4, n=0.875: T_c≈0.118t  →  β_c≈8.5
        #   (Paiva, dos Santos, Scalettar, Denteneer 2004; verify against current paper).
        # Filename uses {alpha:.4f} = |U|. Example: pair_momentum_L8_n0.875_U4.0000-1.csv
        # Each row: [P(q)_features..., β]. n_features = L² for single-orbital square.
        "ALPHAS":            [4.0],
        "FSS_ALPHA":         np.array([4.0]),
        "FSS_BETA_C":        np.array([8.5]),
        "FILE_TEMPLATE":     "pair_momentum_L8_n0.875_U{alpha:.4f}-1.csv",
        "L":                 8,
        "BETA_MIN":          None,
        "BETA_MAX":          None,
        "BETA_MIN_PER_ALPHA": {},
        "BETA_MAX_PER_ALPHA": {},
        # ── Observable registry ──────────────────────────────────────────────
        # Each Hubbard observable lives in its own subdir under
        # DATA_DIR/hubbard_square/csv/.  Set OBSERVABLE in run_scripts/* (or
        # pass observable= to pipeline.run) to one of these names and
        # DATA_DIR + FILE_TEMPLATE are resolved automatically via
        # OBSERVABLE_DATA_PATTERN below.  Layout:
        #   data/hubbard_square/
        #       sim_data/  ← raw per-β DQMC sim folders
        #       csv/<obs>_L8_n0.875_U4.0000-1_csv/<obs>_L8_n0.875_U4.0000-1.csv
        "OBSERVABLES": [
            "density_position", "density_momentum",
            "pair_position",    "pair_momentum",
            "integrated_density_position", "integrated_density_momentum",
            "integrated_pair_position",    "integrated_pair_momentum",
        ],
        "OBSERVABLE_DATA_PATTERN": {
            "data_subdir":   "hubbard_square/csv/{obs}_L8_n0.875_U4.0000-1_csv",
            "file_template": "{obs}_L8_n0.875_U{{alpha:.4f}}-1.csv",
        },
    },
    "bond_holstein": {
        # Beta-sweep files (fixed alpha, varying beta) — TBD when CSVs are produced.
        # Coupling convention: λ_user = α²/2  (x-axis of Fig. 7 is 1/λ_user).
        # Paper's λ_bond = 2·λ_user, so paper's λ_bond=0.4 ↔ α=0.6325 (λ_user=0.2).
        # FSS β_c from BondHolstein_paper.pdf Fig. 7 (green DQMC FSS line).
        # x-axis of Fig. 7 is 1/λ_user where λ_user = α²/2.
        # Our ALPHAS map to exactly x = 5, 4, 3, 2, 1 on Fig. 7.
        # Anchor points: (1/λ=5, T=0.63) explicit (p.5); (1/λ=3.333, T=1.0) Fig.7 read.
        # Values at 1/λ=1,2 are extrapolations — higher uncertainty.
        #   α=0.6325 → 1/λ=5.0 → T=0.63  → β_c=1.58  (explicit)
        #   α=0.7071 → 1/λ=4.0 → T≈0.85  → β_c≈1.18  (interpolated)
        #   α=0.8165 → 1/λ=3.0 → T≈1.07  → β_c≈0.93  (near anchor)
        #   α=1.0000 → 1/λ=2.0 → T≈1.5   → β_c≈0.67  (extrapolated)
        #   α=1.4142 → 1/λ=1.0 → T≈2.0   → β_c≈0.50  (left edge of Fig. 7)
        "ALPHAS":            [0.6325, 0.7071, 0.8165, 1.0000, 1.4142],
        "FSS_ALPHA":         np.array([0.6325, 0.7071, 0.8165, 1.0000, 1.4142]),
        "FSS_BETA_C":        np.array([1.58,   1.18,   0.93,   0.67,   0.50]),
        "FILE_TEMPLATE":     "electron_densities_w1.00_a{alpha:.4f}_L12-1.csv",
        # Alpha-sweep file (fixed beta, varying alpha): use load_alpha_sweep().
        # Template uses {beta:.2f} placeholder.
        # Available: b0.35 → α∈[1.0, 2.5] step 0.1, 1000 samples/α.
        # At β=0.35 (T=2.86t), expected α_c ≈ 1.5–1.7 from Fig. 7 extrapolation.
        "ALPHA_SWEEP_FILE_TEMPLATE": "electron_densities_w1.00_b{beta:.2f}_L12-20.csv",
        "DATA_SUBDIR":       "bond_holstein/csv",
        "L":                 12,
        "BETA_MIN":          None,
        "BETA_MAX":          None,
        "BETA_MIN_PER_ALPHA": {},
        "BETA_MAX_PER_ALPHA": {},
    },
}


def apply_model(name):
    """
    Apply a named model preset to this module's global configuration.

    Call this *before* the method modules compute anything (they access
    config values lazily through the module object).  In run_all.py the
    call happens before the lazy imports, so every method picks up the
    updated values automatically.

    Parameters
    ----------
    name : str
        One of "ssh", "holstein", "bond_holstein".

    Returns
    -------
    dict  — the applied preset (same object as MODEL_CONFIGS[name]).
    """
    import sys
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_CONFIGS)}"
        )
    cfg = MODEL_CONFIGS[name]
    mod = sys.modules[__name__]
    for key, val in cfg.items():
        setattr(mod, key, val)
    # Rebuild derived attributes
    mod.SIZE    = mod.L * mod.L
    mod.FSS_DICT = dict(zip(mod.FSS_ALPHA, mod.FSS_BETA_C))
    # Resolve DATA_DIR from preset's DATA_SUBDIR (joined onto base data/);
    # if the preset has no DATA_SUBDIR, reset to the base data/ directory so
    # switching models doesn't carry over a previous example's path.
    base_data = os.path.join(PROJECT_ROOT, "data")
    sub = cfg.get("DATA_SUBDIR")
    mod.DATA_DIR = os.path.join(base_data, sub) if sub else base_data
    return cfg

# ── Data / observable configuration ───────────────────────────────────────────
# FILE_TEMPLATE: filename template for beta-sweep data (fixed alpha, varying beta).
# Use {alpha:.4f} as the placeholder for the coupling constant.
# Change this (or pass --file-template on the CLI) when using a different observable.
#
# Available observables (all in data/):
#   BondBond order parameter momentum-space (default):
#     "BondBond012345_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"   576 features
#   Green's function momentum-space:
#     "greens0_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"          144 features
#
# Phonon observables (produced by extract_phonons.py after new DQMC runs):
#   Tau=0 phonon field (raw x,y displacements for all 4 modes):
#     "phonon_tau0_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"               576 features
#     Use with: PCA Var(P1) may give a circle (3-domain KVBS).
#     Better: use UMAP or the radial-displacement file below.
#   Radial displacement (τ-averaged |displacement| per site, domain-invariant):
#     "radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"               288 features
#     Recommended for PCA: gives clear 2-cluster structure.
#
# For alpha-sweep data (fixed beta, varying alpha), pass a separate template to
# load_alpha_sweep() with a {beta:.Xf} placeholder instead.
FILE_TEMPLATE = "BondBond012345_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
# FILE_TEMPLATE = "BondBond0_position_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
# FILE_TEMPLATE = "BondBond0_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
# FILE_TEMPLATE = "greens0_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
# FILE_TEMPLATE = "phonon_tau0_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
# FILE_TEMPLATE = "radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"

# BETA_MIN / BETA_MAX: global beta bounds applied to all alphas (None = no limit).
# The bond-bond momentum-space data contains a spurious V-shaped crossover at
# beta ~ 12; setting BETA_MIN ~ 8 removes this low-beta contamination.
BETA_MIN = None
BETA_MAX = None

# Per-alpha beta bounds — looked up FIRST; global BETA_MIN/BETA_MAX are fallbacks.
# Set to None entries (or leave dict empty) to use the global values.
# Leave dicts empty to rely on globals or CLI --beta-min / --beta-max.
#
# Suggested for greens0_momentum (from UMAP fKVBS onset analysis):
#   BETA_MIN_PER_ALPHA = {0.240: 11, 0.245: 8, 0.250: 5, 0.255: 4}
#
# Suggested for BondBond012345_momentum (from V-shape crossover analysis):
#   BETA_MIN_PER_ALPHA = {0.240: 10, 0.245: 8, 0.250: 5, 0.255: 5}
#
BETA_MIN_PER_ALPHA = {}
BETA_MAX_PER_ALPHA = {}

# ── Device selection ──────────────────────────────────────────────────────────
# MPS can cause silent hangs with PyTorch 2.2; use CPU for reliability.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
