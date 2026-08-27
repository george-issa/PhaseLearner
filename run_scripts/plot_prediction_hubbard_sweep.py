#!/usr/bin/env python3
"""
Re-render prediction-based PDFs for the full 2 densities × 8 observables
Hubbard sweep (no recomputation).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

DENSITIES   = [0.500, 0.875]
OBSERVABLES = [
    "density_position", "density_momentum",
    "pair_position",    "pair_momentum",
    "integrated_density_position", "integrated_density_momentum",
    "integrated_pair_position",    "integrated_pair_momentum",
]

for n in DENSITIES:
    for obs in OBSERVABLES:
        print(f"\n{'='*70}\n>>> plot  n={n:.3f}  obs={obs}\n{'='*70}")
        run(
            model      = "attractive_hubbard",
            observable = obs,
            methods    = ["predict"],
            n_target   = n,
            plot_only  = True,
            plot       = True,
        )
