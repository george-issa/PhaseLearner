#!/usr/bin/env python3
"""
Compare PCA / UMAP / t-SNE beta_c across every oSSH observable.

Reads the .npz files written by run_methods_ssh.py and scores each
(method, observable) pair against the FSS reference.

The three methods share two estimators, which is what makes them comparable:

  km_sig    sigmoid midpoint of the ordered-cluster fraction f(beta)
  km_deriv  steepest-slope point of the same curve (fit-free cross-check)

They sit at different indices in each method's saved ``beta_c`` array
(see ESTIMATOR_INDEX below).  PCA additionally carries the Var(P1)
susceptibility estimators, reported separately by --pca-extra since UMAP and
t-SNE have no counterpart.

Scores
------
relRMSE   RMS of (beta_c - FSS)/FSS over the four FSS-anchored alphas.
          Lower is better.  alpha=0.2425 has no FSS value and is excluded.
rise      Largest increase of beta_c between consecutive alphas, in units of
          beta.  The physical boundary falls with alpha, so any rise is a
          trend violation -- but FSS itself is nearly flat over the last step
          (12.5 -> 12.3), so a rise of ~0.2 is noise and only a large one
          (>~1) means the estimator is tracking something else.

Run:  python scripts/compare_methods_ssh.py --sid method_compare
      python scripts/compare_methods_ssh.py --pca-extra
      python scripts/compare_methods_ssh.py --csv results/method_compare_summary.csv
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]
FSS = {0.240: 17.5, 0.245: 14.3, 0.250: 12.5, 0.255: 12.3}

# method -> {estimator: index into the saved beta_c array}
ESTIMATOR_INDEX = {
    "pca":  {"km_sig": 2, "km_deriv": 4},
    "umap": {"km_sig": 0, "km_deriv": 2},
    "tsne": {"km_sig": 0, "km_deriv": 2},
}
# PCA-only Var(P1) susceptibility estimators (no UMAP / t-SNE counterpart).
PCA_EXTRA = {"argmax_P1": 0, "best": 5, "mean_P1_deriv": 6}

METHODS = ["pca", "umap", "tsne"]


def load_betac(npz_path, index):
    """beta_c at `index` for every alpha; NaN where the alpha is missing."""
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    out = []
    for a in ALPHAS:
        key = f"a{a:.4f}_beta_c"
        out.append(float(d[key][index]) if key in d.files else np.nan)
    return np.array(out)


def rel_rmse(vals):
    """RMS relative deviation from FSS over the anchored alphas."""
    err = [(vals[i] - FSS[a]) / FSS[a]
           for i, a in enumerate(ALPHAS) if a in FSS and np.isfinite(vals[i])]
    return np.sqrt(np.mean(np.square(err))) if len(err) == len(FSS) else np.nan


def max_rise(vals):
    """Largest increase of beta_c between consecutive alphas (0.0 = monotone)."""
    v = vals[np.isfinite(vals)]
    return float(max(0.0, np.max(np.diff(v)))) if len(v) > 1 else np.nan


def collect(npz_dir, estimators, methods):
    """One record per (method, observable, estimator) that has a saved file."""
    rows = []
    for method in methods:
        prefix = f"{method}_results_"
        if not os.path.isdir(npz_dir):
            continue
        for fname in sorted(os.listdir(npz_dir)):
            if not (fname.startswith(prefix) and fname.endswith("_sid5.npz")):
                continue
            obs = fname[len(prefix):-len("_sid5.npz")]
            for est, idx in estimators.get(method, {}).items():
                vals = load_betac(os.path.join(npz_dir, fname), idx)
                if vals is None or not np.isfinite(vals).any():
                    continue
                rows.append({
                    "method": method, "observable": obs, "estimator": est,
                    "betac": vals, "relrmse": rel_rmse(vals),
                    "rise": max_rise(vals),
                })
    return rows


def print_table(rows, title, limit=None):
    print(f"\n{title}")
    header = (f"{'method':<7}{'observable':<37}{'estimator':<11}"
              + "".join(f"{a:>8.4f}" for a in ALPHAS)
              + f"{'relRMSE':>10}{'rise':>7}")
    print(header)
    print("-" * len(header))
    print(f"{'FSS':<7}{'reference':<37}{'':<11}"
          + "".join(f"{FSS.get(a, float('nan')):>8.2f}" for a in ALPHAS))
    for r in rows[:limit]:
        print(f"{r['method']:<7}{r['observable']:<37}{r['estimator']:<11}"
              + "".join(f"{v:>8.2f}" for v in r["betac"])
              + f"{r['relrmse']:>10.4f}{r['rise']:>7.2f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sid", default="method_compare",
                    help="results/<sid>/npz/ to read (default: method_compare)")
    ap.add_argument("--top", type=int, default=25,
                    help="rows in the overall ranking (default: 25)")
    ap.add_argument("--pca-extra", action="store_true",
                    help="also score the PCA-only Var(P1) estimators")
    ap.add_argument("--csv", default=None, help="write the full ranking here")
    args = ap.parse_args()

    npz_dir = os.path.join(ROOT, "results", args.sid, "npz")
    estimators = {m: dict(v) for m, v in ESTIMATOR_INDEX.items()}
    if args.pca_extra:
        estimators["pca"].update(PCA_EXTRA)

    rows = collect(npz_dir, estimators, METHODS)
    if not rows:
        sys.exit(f"no result files under {npz_dir}")

    scored = [r for r in rows if np.isfinite(r["relrmse"])]
    scored.sort(key=lambda r: r["relrmse"])

    print(f"Read {len(rows)} (method, observable, estimator) records "
          f"from {npz_dir}")
    print_table(scored, f"=== Overall ranking (top {args.top}) ===", args.top)

    # Best observable for each method, on each shared estimator.
    print("\n=== Best observable per method ===")
    print(f"{'method':<7}{'estimator':<11}{'observable':<37}"
          f"{'relRMSE':>10}{'rise':>7}")
    for method in METHODS:
        for est in ("km_sig", "km_deriv"):
            cand = [r for r in scored
                    if r["method"] == method and r["estimator"] == est]
            if not cand:
                print(f"{method:<7}{est:<11}(no results)")
                continue
            b = cand[0]
            print(f"{method:<7}{est:<11}{b['observable']:<37}"
                  f"{b['relrmse']:>10.4f}{b['rise']:>7.2f}")

    # Head-to-head: which method wins on each observable (km_sig).
    print("\n=== Winning method per observable (km_sig) ===")
    by_obs = {}
    for r in scored:
        if r["estimator"] == "km_sig":
            by_obs.setdefault(r["observable"], []).append(r)
    tally = {}
    print(f"{'observable':<37}" + "".join(f"{m:>10}" for m in METHODS)
          + f"{'winner':>10}")
    print("-" * 77)
    for obs in sorted(by_obs):
        cand = sorted(by_obs[obs], key=lambda r: r["relrmse"])
        scores = {r["method"]: r["relrmse"] for r in cand}
        win = cand[0]["method"]
        tally[win] = tally.get(win, 0) + 1
        print(f"{obs:<37}"
              + "".join(f"{scores.get(m, float('nan')):>10.4f}" for m in METHODS)
              + f"{win:>10}")
    print("\nwins: " + ", ".join(f"{m}={tally.get(m, 0)}" for m in METHODS))

    if args.csv:
        path = args.csv if os.path.isabs(args.csv) else os.path.join(ROOT, args.csv)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("rank,method,observable,estimator,"
                    + ",".join(f"betac_a{a:.4f}" for a in ALPHAS)
                    + ",relrmse,max_rise\n")
            for i, r in enumerate(scored, 1):
                f.write(f"{i},{r['method']},{r['observable']},{r['estimator']},"
                        + ",".join(f"{v:.3f}" for v in r["betac"])
                        + f",{r['relrmse']:.4f},{r['rise']:.3f}\n")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
