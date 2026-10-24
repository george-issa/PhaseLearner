#!/usr/bin/env python3
"""
Prediction-based β_c sweep: 2 densities × 8 observables on the attractive
Hubbard pilot data. Mirrors the layout of the existing PCA + UMAP sweep so
results land in results/npz/predict_results_{obs}_n{n}.npz.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

DENSITIES   = [0.500, 0.875]
OBSERVABLES = [
    "density_position", "density_momentum",
    "pair_position",    "pair_momentum",
    "integrated_density_position", "integrated_density_momentum",
    "integrated_pair_position",    "integrated_pair_momentum",
]
N_SEEDS = 5    # Greplova-style ensembling

summary = []  # (n_target, obs, bc_nn, bc_mean, dt)

from ssh_ml_phase.data import result_filename
from ssh_ml_phase import config as _cfg

for n in DENSITIES:
    for obs in OBSERVABLES:
        print(f"\n{'='*70}\n>>> n={n:.3f}  obs={obs}\n{'='*70}")
        expected_tag = f"{obs}_n{n:.3f}"
        nn_path   = os.path.join(_cfg.NPZ_DIR, result_filename("predict", expected_tag))
        mean_path = os.path.join(_cfg.NPZ_DIR, result_filename("mean",    expected_tag))

        needs_nn   = not os.path.exists(nn_path)
        needs_mean = not os.path.exists(mean_path)
        methods    = ([m for m, need in zip(("predict", "mean"),
                                            (needs_nn, needs_mean)) if need])

        if not methods:
            print(f"  (NN + mean both already computed → skipping)")
            dt = 0.0
        else:
            t0 = time.time()
            run(
                model      = "attractive_hubbard",
                observable = obs,
                methods    = methods,
                n_target   = n,
                n_seeds    = N_SEEDS,
                plot       = True,
            )
            dt = time.time() - t0

        # Load β_c from both files for the summary
        bc_nn   = float(np.load(nn_path)["a4.0000_beta_c"][0]) \
                  if os.path.exists(nn_path) else float("nan")
        bc_mean = float(np.load(mean_path)["a4.0000_beta_c"][0]) \
                  if os.path.exists(mean_path) else float("nan")
        summary.append((n, obs, bc_nn, bc_mean, dt))

print("\n" + "=" * 70)
print("SUMMARY  —  prediction-based β_c (Hubbard, |U|=4)")
print("=" * 70)
print(f"{'n':>6}  {'observable':<32}  {'β_c^NN':>8}  {'β_c^mean':>9}  "
      f"{'Δ':>5}  {'dt(s)':>7}")
print("-" * 78)
for n, obs, bc_nn, bc_mean, dt in summary:
    if not np.isnan(bc_mean):
        gap = abs(bc_nn - bc_mean)
        gap_str = f"{gap:.2f}"
    else:
        gap_str = "--"
    print(f"{n:>6.3f}  {obs:<32}  {bc_nn:>8.2f}  {bc_mean:>9.2f}  "
          f"{gap_str:>5}  {dt:>7.1f}")
print("\nReference (from [[03_Source_Map#Attractive Hubbard T_c reference]]):")
print("  n=0.500   FSS β_c band [6.54, 7.51]  (Paiva 2004 PRG plateau)")
print("  n=0.875   FSS β_c band [6.54, 7.51]  (same plateau anchor)")
