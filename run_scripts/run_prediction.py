#!/usr/bin/env python3
"""
Run the prediction-based method (NN regresses β from configurations).

Edit the USER CONFIG block, then::

    python run_scripts/run_prediction.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

# ── USER CONFIG ──────────────────────────────────────────────────────────────
MODEL      = "attractive_hubbard"            # ssh | holstein | bond_holstein | attractive_hubbard
OBSERVABLE = "pair_momentum"                 # see config.py for valid names per model

ALPHAS     = None        # None → use the model preset's full list
BETA_MIN   = None        # discard samples with β < this
BETA_MAX   = None        # discard samples with β > this
STRIDE     = None        # use every N-th row (e.g. 10 for 1000-bin data)
TAG        = None        # output label; None → auto-derived from FILE_TEMPLATE
N_TARGET   = None        # override preset's ⟨n⟩ (e.g. 0.5); None → use preset default
N_SEEDS    = 3           # number of random seeds to average over

PLOT       = False       # render PDFs after computing
# ─────────────────────────────────────────────────────────────────────────────


run(
    model        = MODEL,
    observable   = OBSERVABLE,
    alphas       = ALPHAS,
    beta_min     = BETA_MIN,
    beta_max     = BETA_MAX,
    stride       = STRIDE,
    tag          = TAG,
    methods      = ["predict"],
    plot         = PLOT,
    n_target     = N_TARGET,
    n_seeds      = N_SEEDS,
)
