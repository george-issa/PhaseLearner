#!/usr/bin/env python3
"""PCA order-parameter-fluctuation estimator of beta_c on the raw phonon field.

For each alpha: load phonon_tau0 (576 numbers per configuration, 1000 per beta),
centre, PCA over the whole beta sweep.  The two leading components are degenerate
and span the Kekule order-parameter plane; the symmetry-invariant amplitude is the
radius r = |(P1, P2)| (the analogue of |m| for Ising).  beta* = position of the
peak of Var(r) across samples (Gaussian-smoothed, sigma = 1 grid step) -- the
susceptibility of the learned order parameter.

Central value and error: the peak position is recomputed on bootstrap resamples
of the configurations at every beta; beta_c is the bootstrap MEDIAN and the bar
is its 16-84 % range, so the point sits inside its own bar by construction (the
full-data peak is stored too, as beta_peak_full).  The stored bins are
statistically independent (integrated autocorrelation time of r ~ 0.5 bins,
2026-09-06 check), so no block or walker structure is needed.  A second,
systematic spread is reported from varying the smoothing width over
sigma = 0.5, 1, 2 grid steps (beta_sys_lo / beta_sys_hi).

Validated 2026-09-06 (obsidian/06_Experiments.md): interior peak, unchanged under
truncation of the beta window, monotone in alpha, finite-size shift in the
expected direction.

Output: results/npz/pca_var_r_<observable>.npz with alphas, beta_c, beta_lo,
beta_hi, n_boot -- read by paper/scripts/fig_phase_diagram.py.

Usage (repo root):  python run_scripts/pca_var_r_peak.py [--n-boot 200]
"""
import argparse
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ssh_ml_phase.data import load_beta_sweep  # noqa: E402

ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]
FSS = {0.240: 17.5, 0.245: 14.3, 0.250: 12.5, 0.255: 12.3}
DATA_DIR = os.path.join(ROOT, "data", "ssh", "csv")


def var_r_peak(r, betas, u, rng, n_boot, sigma=1.0):
    """Full-data peak, bootstrap median and 16/84 % of the smoothed Var(r) peak,
    plus the min/max of the full-data peak over sigma = 0.5, 1, 2 grid steps."""
    groups = [r[betas == b] for b in u]
    var = np.array([g.var() for g in groups])
    peak_full = u[np.argmax(gaussian_filter1d(var, sigma))]
    boots = []
    for _ in range(n_boot):
        vb = np.array([g[rng.integers(0, len(g), len(g))].var() for g in groups])
        boots.append(u[np.argmax(gaussian_filter1d(vb, sigma))])
    lo, med, hi = np.percentile(boots, [16, 50, 84])
    sys_peaks = [u[np.argmax(gaussian_filter1d(var, s_))] for s_ in (0.5, 1.0, 2.0)]
    return peak_full, med, lo, hi, min(sys_peaks), max(sys_peaks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observable", default="phonon_tau0")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'alpha':>7s} {'median':>6s} {'lo':>5s} {'hi':>5s} {'full':>5s} {'sys':>9s}  FSS   explained var PC1+PC2")
    for a in ALPHAS:
        X, betas = load_beta_sweep(a, data_dir=DATA_DIR, transform=None,
                                   file_template=args.observable + "_L12_w0.5_m0.0_a{alpha:.4f}-5.csv")
        X = X - X.mean(0)
        pca = PCA(n_components=2).fit(X)
        P = pca.transform(X)
        r = np.hypot(P[:, 0], P[:, 1])
        u = np.unique(betas)
        full, med, lo, hi, slo, shi = var_r_peak(r, betas, u, rng, args.n_boot)
        rows.append((a, med, lo, hi, full, slo, shi))
        ev = pca.explained_variance_ratio_
        print(f"{a:7.4f} {med:6.1f} {lo:5.1f} {hi:5.1f} {full:5.1f} {slo:4.1f}-{shi:4.1f}  {FSS.get(a, float('nan')):4.1f}   {ev[0]:.3f}+{ev[1]:.3f}")
    r_ = np.array(rows)
    out = os.path.join(ROOT, "results", "npz", f"pca_var_r_{args.observable}.npz")
    np.savez(out, alphas=r_[:, 0], beta_c=r_[:, 1], beta_lo=r_[:, 2], beta_hi=r_[:, 3],
             beta_peak_full=r_[:, 4], beta_sys_lo=r_[:, 5], beta_sys_hi=r_[:, 6], n_boot=args.n_boot)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
