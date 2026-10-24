"""
Verify that each of the 8 attractive-Hubbard CSV folders is well-shaped
for every ML method in the pipeline:

  PCA, UMAP             — observable-agnostic; need a 2-D (N, F) matrix.
  Autoencoder           — needs (N, F); reads F at runtime.
  Assaad/GI CNN         — needs F == L² where L = config.L (so the
                          reshape(N, 1, L, L) succeeds and the two
                          MaxPool2d(2) layers leave a (L//4)² spatial map).

For each folder we also report:
  n_rows         — total samples
  n_betas        — distinct β values
  rows / β       — samples per β (DQMC bin count)
  beta range     — min..max
"""
import os
import numpy as np
from ssh_ml_phase import config as _cfg
_cfg.apply_model("attractive_hubbard")          # L=8 for Hubbard

OBSERVABLES = [
    "density_position", "density_momentum",
    "pair_position",    "pair_momentum",
    "integrated_density_position", "integrated_density_momentum",
    "integrated_pair_position",    "integrated_pair_momentum",
]

L      = _cfg.L
NEEDED = L * L
DATA_ROOT = os.path.join(_cfg.DATA_DIR, "hubbard_square", "csv")

print(f"Model: attractive_hubbard   L = {L}   F_required = L² = {NEEDED}")
print(f"Data root: {DATA_ROOT}")
print("-" * 110)
print(f"{'observable':<32}  {'rows':>6}  {'F':>4}  {'n_β':>4}  {'rows/β':>7}  "
      f"{'β-range':<14}  PCA  UMAP  AE   CNN")
print("-" * 110)

all_ok = True
for obs in OBSERVABLES:
    folder = os.path.join(DATA_ROOT, f"{obs}_L{L}_n0.875_U4.0000-1_csv")
    giant  = os.path.join(folder, f"{obs}_L{L}_n0.875_U4.0000-1.csv")

    if not os.path.isfile(giant):
        print(f"{obs:<32}  MISSING giant CSV: {giant}")
        all_ok = False
        continue

    a       = np.loadtxt(giant, delimiter=",")
    n_rows  = a.shape[0]
    F       = a.shape[1] - 1
    betas   = a[:, -1]
    u_beta  = np.unique(betas)
    counts  = [(betas == b).sum() for b in u_beta]
    rpb     = counts[0] if len(set(counts)) == 1 else f"{min(counts)}–{max(counts)}"
    brange  = f"{u_beta.min():.1f}…{u_beta.max():.1f}"

    pca_ok  = "✓"
    umap_ok = "✓"
    ae_ok   = "✓"
    cnn_ok  = "✓" if F == NEEDED and (L % 4 == 0) else "✗"

    if cnn_ok != "✓":
        all_ok = False

    print(f"{obs:<32}  {n_rows:>6}  {F:>4}  {len(u_beta):>4}  {rpb!s:>7}  "
          f"{brange:<14}  {pca_ok:>3}  {umap_ok:>4}  {ae_ok:>3}  {cnn_ok:>4}")

print("-" * 110)
print("All folders OK." if all_ok else "*** Some folders have shape problems ***")
