#!/usr/bin/env python3
"""
Retrain the prediction-based regressor on the 4 pair-correlator observables
(static + integrated, position + momentum) at both densities, recording
loss curves and early-stop metadata for convergence diagnostics.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

DENSITIES   = [0.500, 0.875]
OBSERVABLES = [
    "pair_position",
    "pair_momentum",
    "integrated_pair_position",
    "integrated_pair_momentum",
]
N_SEEDS = 3

for n in DENSITIES:
    for obs in OBSERVABLES:
        print(f"\n{'='*70}\n>>> n={n:.3f}  obs={obs}\n{'='*70}")
        t0 = time.time()
        run(
            model      = "attractive_hubbard",
            observable = obs,
            methods    = ["predict"],
            n_target   = n,
            n_seeds    = N_SEEDS,
            plot       = True,
        )
        print(f"  (run+plot took {time.time() - t0:.1f}s)")

# ── Summary: final val loss + R² per (n, observable) ────────────────────────
from ssh_ml_phase.data import result_filename
from ssh_ml_phase import config as _cfg

print("\n" + "=" * 78)
print("CONVERGENCE SUMMARY  —  best val MSE on z-scored β (R² = 1 − MSE)")
print("=" * 78)
print(f"{'n':>6}  {'observable':<28}  {'val MSE':>14}  {'R²':>14}  {'epoch':>10}")
print("-" * 78)
for n in DENSITIES:
    for obs in OBSERVABLES:
        tag  = f"{obs}_n{n:.3f}"
        path = os.path.join(_cfg.NPZ_DIR, result_filename("predict", tag))
        z    = np.load(path)
        bv   = z["a4.0000_best_val_per_seed"]
        be   = z["a4.0000_best_epoch_per_seed"]
        nep  = z["a4.0000_n_epochs_per_seed"]
        med_val = float(np.median(bv))
        r2      = 1.0 - med_val
        bv_str  = "/".join(f"{v:.3f}" for v in bv)
        be_str  = "/".join(str(int(e))  for e in be)
        nep_str = "/".join(str(int(e))  for e in nep)
        print(f"{n:>6.3f}  {obs:<28}  "
              f"{bv_str:>14}  {r2:>13.3f}*  {be_str:>4}/{nep_str:<5}")
print("\n* R² shown is on the median seed (z-scored val MSE; 1 = perfect)")
print("  'epoch' column = best_epoch / total_epochs per seed")
