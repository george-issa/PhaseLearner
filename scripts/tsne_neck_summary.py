#!/usr/bin/env python3
"""Collect the t-SNE NECK beta_c over an ensemble of runs into one summary .npz.

Each run is a results/npz/tsne_results_<tag>.npz written by ssh_ml_phase.tsne_method
(the neck is recomputed from each run's saved ``a<alpha>_neck_curve`` with
``tsne_method.neck_from_curve`` so every run obeys the same prominence rule).  Runs differ by t-SNE perplexity
and/or random seed; the ensemble spread is the reported uncertainty.

Output: results/npz/tsne_neck_<observable>.npz with arrays
    alphas, beta_c (ensemble median), beta_lo, beta_hi (ensemble min / max),
    depth (median neck depth), n_runs
which paper/scripts/fig_phase_diagram.py reads.

Usage (repo root):
    python scripts/tsne_neck_summary.py BondBond012345_momentum \\
        --tags BondBond012345_momentum_neck_p30 BondBond012345_momentum_neck_p100 ...
"""
import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_DIR = os.path.join(ROOT, "results", "npz")
FSS = {0.240: 17.5, 0.245: 14.3, 0.250: 12.5, 0.255: 12.3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observable")
    ap.add_argument("--tags", nargs="+", required=True, help="tags of the tsne_results_<tag>.npz runs")
    ap.add_argument("--min-depth", type=float, default=0.8,
                    help="ignore a run's neck if its depth is above this (no real dip)")
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    from ssh_ml_phase.tsne_method import neck_from_curve   # one rule for every run

    necks = {}                                   # alpha -> list of (neck, depth, tag)
    for tag in args.tags:
        path = os.path.join(NPZ_DIR, f"tsne_results_{tag}.npz")
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr); continue
        d = np.load(path, allow_pickle=True)
        for key in d.files:
            m = re.fullmatch(r"a(\d\.\d+)_neck_curve", key)
            if m:
                a = m.group(1)
                beta, depth = neck_from_curve(d[f"a{a}_u_beta"], d[key])
                necks.setdefault(round(float(a), 4), []).append((beta, depth, tag))

    alphas = sorted(necks)
    rows = []
    print(f"{'alpha':>7s} {'beta_c':>7s} {'lo':>5s} {'hi':>5s} {'depth':>6s} {'runs':>4s}  FSS   per-run necks")
    for a in alphas:
        good = [(b, dp, t) for b, dp, t in necks[a] if np.isfinite(b) and dp <= args.min_depth]
        if not good:
            print(f"{a:7.4f}  no run with a prominent dip ({len(necks[a])} runs)"); continue
        b = np.array([g[0] for g in good]); dp = np.array([g[1] for g in good])
        rows.append((a, np.median(b), b.min(), b.max(), np.median(dp), len(b)))
        print(f"{a:7.4f} {np.median(b):7.2f} {b.min():5.1f} {b.max():5.1f} {np.median(dp):6.2f} {len(b):4d}  "
              f"{FSS.get(a, float('nan')):4.1f}   " + ", ".join(f"{g[0]:.1f}({g[2].split('_')[-1]})" for g in good))
    out = os.path.join(NPZ_DIR, f"tsne_neck_{args.observable}.npz")
    r = np.array(rows)
    np.savez(out, alphas=r[:, 0], beta_c=r[:, 1], beta_lo=r[:, 2], beta_hi=r[:, 3],
             depth=r[:, 4], n_runs=r[:, 5].astype(int), tags=np.array(args.tags))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
