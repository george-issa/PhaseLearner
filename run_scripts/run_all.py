#!/usr/bin/env python3
"""
Run every requested ML method (PCA + UMAP by default), then plot.

Edit the USER CONFIG block, then::

    python run_scripts/run_all.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

# ── USER CONFIG ──────────────────────────────────────────────────────────────
MODEL      = "attractive_hubbard"            # ssh | holstein | bond_holstein | attractive_hubbard
OBSERVABLE = "integrated_pair_position"      # see config.py for valid names per model

ALPHAS     = None        # None → use the model preset's full list
BETA_MIN   = None        # discard samples with β < this
BETA_MAX   = None        # discard samples with β > this
STRIDE     = None        # use every N-th row (e.g. 10 for 1000-bin data)
TAG        = None        # output label; None → auto-derived from FILE_TEMPLATE

# Subset of {"pca", "umap", "assaad", "ae"}.  Assaad and AE are expensive.
METHODS    = ["pca", "umap"]

PLOT       = True        # render PDFs after computing
PLOT_ONLY  = False       # True → skip compute, just re-render PDFs from existing .npz

# Alpha-sweep mode (fixed β, varying α).  Most users leave this alone.
ALPHA_SWEEP  = False
BETAS        = None      # e.g. [0.35]
FSS_ALPHA_C  = None      # FSS reference α_c for sigmoid fit init
# ─────────────────────────────────────────────────────────────────────────────


run(
    model        = MODEL,
    observable   = OBSERVABLE,
    alphas       = ALPHAS,
    beta_min     = BETA_MIN,
    beta_max     = BETA_MAX,
    stride       = STRIDE,
    tag          = TAG,
    methods      = METHODS,
    plot         = PLOT,
    plot_only    = PLOT_ONLY,
    alpha_sweep  = ALPHA_SWEEP,
    betas        = BETAS,
    fss_alpha_c  = FSS_ALPHA_C,
)
