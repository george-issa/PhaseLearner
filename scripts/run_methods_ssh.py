#!/usr/bin/env python3
"""
Run PCA + UMAP + t-SNE on every oSSH observable (sID-5 1000-bin runs).

One sweep, one output folder, so the three methods can be compared on
identical inputs.  Sibling of run_pca_ssh_all_obs.py / run_umap_ssh_all_obs.py,
which run a single method each.

Inputs:  data/ssh/csv/<obs>_L12_w0.5_m0.0_a{alpha:.4f}-5.csv
         (built by create_csv_ssh_all_obs.py + extract_phonons_edited.py)
Outputs: results/<sid>/npz/{pca,umap,tsne}_results_<obs>_sid5.npz

The default --sid keeps this sweep out of the top-level results/npz/, so the
existing paper-figure inputs are not touched.

Observables (34): every bond id-pair (eq-time + integrated, position +
momentum), every greens id-pair (eq-time only, both spaces), the KVBS
combination (momentum only, eq-time + integrated), and the two phonon
channels.  No beta floor is applied — see the vault note on floors being
biased when chosen from structure adjacent to the transition.

Run:  python scripts/run_methods_ssh.py                    # all 34, all 3 methods
      python scripts/run_methods_ssh.py --workers 6
      python scripts/run_methods_ssh.py --methods pca tsne
      python scripts/run_methods_ssh.py integrated_bond33_momentum
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Alphas with sID-5 data on disk.
ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]

BOND_PAIRS = ["33", "44", "55", "34", "35", "45"]
GREENS_PAIRS = ["11", "22", "12"]
SPACES = ["position", "momentum"]
PHONON = ["phonon_tau0", "radial_disp"]

DEFAULT_SID = "method_compare"


def all_observables():
    """The 32 correlator observables plus the 2 phonon channels."""
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
    return obs + PHONON


def run_one(job):
    """Run the requested methods for a single observable. Returns a status line."""
    obs, methods, sid = job
    # Imported here so each worker process gets its own module-level config.
    from ssh_ml_phase.pipeline import run

    t0 = time.time()
    try:
        run(model="ssh", alphas=ALPHAS,
            file_template=f"{obs}_L12_w0.5_m0.0_a{{alpha:.4f}}-5.csv",
            tag=f"{obs}_sid5", methods=methods, sid=sid, plot=False)
        return f"ok    {obs:<38} {time.time() - t0:6.0f}s"
    except Exception as exc:                      # keep the sweep going
        return f"FAIL  {obs:<38} {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observables", nargs="*",
                    help="subset of observable names (default: all 34)")
    ap.add_argument("--methods", nargs="+", default=["pca", "umap", "tsne"],
                    help="methods to run (default: pca umap tsne)")
    ap.add_argument("--sid", default=DEFAULT_SID,
                    help=f"results/<sid>/ output folder (default: {DEFAULT_SID})")
    ap.add_argument("--workers", type=int, default=6,
                    help="observables to process in parallel (default: 6)")
    ap.add_argument("--threads", type=int, default=3,
                    help="BLAS/numba threads per worker (default: 3)")
    args = ap.parse_args()

    observables = args.observables or all_observables()

    # Cap per-worker threading before the workers import numpy/numba, otherwise
    # 6 workers x 18 threads oversubscribes the machine and everything crawls.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(args.threads)

    print(f"{len(observables)} observable(s) x {args.methods}, "
          f"alphas={ALPHAS}\n"
          f"{args.workers} workers x {args.threads} threads -> results/{args.sid}/npz/\n")

    jobs = [(obs, args.methods, args.sid) for obs in observables]
    t0 = time.time()

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, line in enumerate(pool.imap_unordered(run_one, jobs), 1):
                print(f"[{i}/{len(jobs)}] {line}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {run_one(job)}", flush=True)

    print(f"\nSweep finished in {(time.time() - t0) / 60:.1f} min. "
          f"Compare with: python scripts/compare_methods_ssh.py --sid {args.sid}")


if __name__ == "__main__":
    main()
