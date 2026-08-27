#!/usr/bin/env python3
"""
Run UMAP β_c extraction on oSSH observables (sID-5 1000-bin runs).

UMAP sibling of run_pca_ssh_all_obs.py — same observable registry, same
sID-5 data, same `<obs>_sid5` tags, so the npz / figures sit alongside the
PCA ones without clobbering them (umap_results_* vs pca_results_*).

Inputs:  data/ssh/csv/<obs>_L12_w0.5_m0.0_a{alpha:.4f}-5.csv
         (built by create_csv_ssh_all_obs.py from the JLD2 sim data)
Outputs: results/npz/umap_results_<obs>_sid5.npz                 (one per observable)
         results/figures/ssh/umap/<obs>_sid5/{avg_analysis,embedding,...}.pdf
            = multi-page PDFs, one page per alpha (all 5 alphas in each file)

The full observable set (32) mirrors create_csv_ssh_all_obs.py:
  * each bond id-pair separately, eq-time + integrated, position + momentum
  * each greens id-pair separately, eq-time only, position + momentum
  * the KVBS combination, momentum only, eq-time + integrated

Run:  python run_scripts/run_umap_ssh_all_obs.py --integrated-bonds  # the 13 integrated bond-bond obs
      python run_scripts/run_umap_ssh_all_obs.py                     # compute all 32
      python run_scripts/run_umap_ssh_all_obs.py bond33_momentum     # a subset
      python run_scripts/run_umap_ssh_all_obs.py --integrated-bonds --plot
      python run_scripts/run_umap_ssh_all_obs.py --integrated-bonds --plot-only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run

# Alphas with sID-5 data on disk.
ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]

# Beta window (None = use everything).  Set BETA_MIN ~ 8 to cut the low-β
# V-shape artifact region if it contaminates a given channel.
BETA_MIN = None
BETA_MAX = None

BOND_PAIRS = ["33", "44", "55", "34", "35", "45"]
GREENS_PAIRS = ["11", "22", "12"]
SPACES = ["position", "momentum"]


def all_observables():
    """The 32 observables produced by create_csv_ssh_all_obs.py."""
    obs = []
    for prefix in ("", "integrated_"):
        for ab in BOND_PAIRS:
            for space in SPACES:
                obs.append(f"{prefix}bond{ab}_{space}")
    for ab in GREENS_PAIRS:
        for space in SPACES:
            obs.append(f"greens{ab}_{space}")
    obs.append("BondBond012345_momentum")
    obs.append("integrated_BondBond012345_momentum")
    return obs


def integrated_bond_observables():
    """The 13 integrated bond-bond correlators (per-pair + KVBS combo)."""
    obs = [f"integrated_bond{ab}_{space}"
           for ab in BOND_PAIRS for space in SPACES]
    obs.append("integrated_BondBond012345_momentum")
    return obs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observables", nargs="*",
                    help="subset of observable names (default: all 32, or "
                         "the 13 integrated bond obs with --integrated-bonds)")
    ap.add_argument("--integrated-bonds", action="store_true",
                    help="run the 13 integrated bond-bond observables")
    ap.add_argument("--plot", action="store_true",
                    help="also render UMAP PDFs after computing each observable")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip UMAP compute; only re-render PDFs from existing npz")
    args = ap.parse_args()

    if args.observables:
        observables = args.observables
    elif args.integrated_bonds:
        observables = integrated_bond_observables()
    else:
        observables = all_observables()

    mode = ("plot-only" if args.plot_only
            else "UMAP + plot" if args.plot else "UMAP")
    print(f"[{mode}] {len(observables)} observable(s), alphas={ALPHAS}\n")

    for i, obs in enumerate(observables, 1):
        template = f"{obs}_L12_w0.5_m0.0_a{{alpha:.4f}}-5.csv"
        print(f"\n########## [{i}/{len(observables)}] {obs} ##########")
        # plot=True/plot_only=True render the pipeline's native layout:
        # multi-page PDFs (one page per alpha, all 5 alphas) under
        # results/figures/ssh/umap/<obs>_sid5/.
        run(model="ssh", alphas=ALPHAS, file_template=template,
            tag=f"{obs}_sid5", beta_min=BETA_MIN, beta_max=BETA_MAX,
            methods=["umap"], plot=args.plot or args.plot_only,
            plot_only=args.plot_only)

    print(f"\nAll done. {len(observables)} npz file(s) in results/npz/ "
          f"(umap_results_<obs>_sid5.npz).")
    if args.plot or args.plot_only:
        print("Figures: results/figures/ssh/umap/<obs>_sid5/*.pdf "
              "(all alphas per file)")


if __name__ == "__main__":
    main()
