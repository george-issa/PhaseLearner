"""
Shared configuration: FSS reference values, paths, and physical constants.
"""
import numpy as np
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ── Physical parameters ───────────────────────────────────────────────────────
L = 12
SIZE = L * L  # 144 unit cells; phonon observables have 576 (tau0) or 288 (radial) features

# All alpha values — original 4 + new additions (see RECOMMENDATIONS.txt)
ALPHAS = [0.235, 0.240, 0.2415, 0.2435, 0.245, 0.250, 0.255]

# FSS reference values (w=0.5, from sohan_phase_diagram.pdf)
# New alphas 0.235/0.2415/0.2435: beta_c TBD from FSS once 1000-bin data arrive.
FSS_ALPHA  = np.array([0.240,  0.245,  0.250,  0.255])
FSS_BETA_C = np.array([17.5,   14.3,   12.5,   12.3])

# Mapping from alpha to FSS beta_c (NaN returned for alphas without FSS reference)
FSS_DICT = dict(zip(FSS_ALPHA, FSS_BETA_C))

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
