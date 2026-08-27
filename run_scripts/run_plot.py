#!/usr/bin/env python3
"""
Re-render PDFs from existing .npz results — no recomputation.

Edit the USER CONFIG block, then::

    python run_scripts/run_plot.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

# ── USER CONFIG ──────────────────────────────────────────────────────────────
MODEL      = "attractive_hubbard"            # ssh | holstein | bond_holstein | attractive_hubbard
OBSERVABLE = "integrated_pair_position"      # see config.py for valid names per model

ALPHAS     = None        # None → use the model preset's full list
BETA_MIN   = None        # must match what was passed when computing the .npz
BETA_MAX   = None
STRIDE     = None
TAG        = None        # output label; None → auto-derived from FILE_TEMPLATE
N_TARGET   = None        # override preset's ⟨n⟩ (e.g. 0.5); None → use preset default
SID        = None        # output-folder ID → results/<SID>/{npz,figures}; must
                         # match the SID the .npz were computed under (None → results/)

# Methods whose .npz to plot. Drop "umap" if you only ran PCA, etc.
METHODS    = ["pca", "umap"]
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
    plot_only    = True,
    n_target     = N_TARGET,
    sid          = SID,
)
