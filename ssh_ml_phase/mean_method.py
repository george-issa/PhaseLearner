"""
Mean-based phase-detection method (Arnold et al. 2021, arXiv:2010.04730).

Theoretical basis
-----------------
Arnold et al. prove that the MSE-optimal regressor for the prediction-based
method is

    p̂_opt(x) = Σ_i  P_i(x) p_i  /  Σ_i  P_i(x),

i.e. the NN is computing a weighted average of true parameters.  The
divergence signal `d⟨p̂⟩/dp` that peaks at a transition is therefore
equivalent — in the optimal-MSE limit — to detecting where the per-β
mean of the input feature changes most rapidly.

Recipe
------
For each unique β in the sweep:

    x̄(β) = (1/N_β) Σ_{i: β_i = β}  X_i              # per-β mean feature

Then the divergence signal is

    Δ(β_mid) = || x̄(β + Δβ)  −  x̄(β − Δβ) ||_2,

and β_c = argmax_β Δ(β).  Raw argmax — no smoothing, no NN, no seeds.

Why it's a useful cross-check
-----------------------------
Failure modes are different from the NN's:
  * High-R² regime where the NN's predicted curve is nearly the identity
    line (derivative noisy → spurious argmax) doesn't affect this method.
  * No overfitting, no hyperparameter sensitivity, no seed variance.
  * Cost: O(N · n_features) — milliseconds per observable.
Agreement with the NN method → high confidence in β_c.
Disagreement → flag the observable.

Usage
-----
    python3.11 -m ssh_ml_phase.mean_method --model attractive_hubbard
    python3.11 -m ssh_ml_phase.mean_method --alphas 4.0 --tag pair_momentum
"""
import argparse
import numpy as np

from . import config as _cfg
from .data import (load_beta_sweep, ensure_npz_dir, make_tag, result_filename,
                   lattice_snapshots, store_snapshots)


def compute_mean_method(alpha, beta_min=None, beta_max=None, stride=None,
                        file_template=None):
    """Compute the mean-based divergence Δ(β) and its argmax β_c.

    Returns
    -------
    dict with keys:
        u_beta        : (n_beta,)            unique β values
        x_bar         : (n_beta, n_features) per-β mean feature
        delta         : (n_beta - 2,)        central-difference norm
        beta_mid      : (n_beta - 2,)        β at the midpoints (= u_beta[1:-1])
        beta_c        : float                argmax of Δ
        n_features    : int
        n_samples     : int
    """
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    u_beta     = np.unique(betas)
    fss_bc     = _cfg.FSS_DICT.get(alpha, float(np.mean(u_beta)))
    n_features = X.shape[1]

    # Standardize features (per-feature z-score) so the L2 norm doesn't get
    # dominated by features with the largest absolute magnitudes.  Matches
    # the input convention used by prediction_method.py.
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    # Per-β mean input vector
    x_bar = np.stack([X_norm[betas == b].mean(axis=0) for b in u_beta])

    # Central finite difference of x̄ across β, then L2 norm
    delta    = np.linalg.norm(x_bar[2:] - x_bar[:-2], axis=1)
    beta_mid = u_beta[1:-1]

    idx = int(np.argmax(delta))
    bc  = float(beta_mid[idx])

    inlat_betas, inlat_grids = lattice_snapshots(X, betas)

    return {
        "u_beta":     u_beta,
        "inlat_betas": inlat_betas,
        "inlat_grids": inlat_grids,
        "x_bar":      x_bar.astype(np.float32),
        "delta":      delta,
        "beta_mid":   beta_mid,
        "beta_c":     bc,
        "fss_bc":     fss_bc,
        "n_features": n_features,
        "n_samples":  len(X),
    }


def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None):
    """Compute mean-based β_c for all alphas and save to results/npz/.

    β_c array layout: ``[deriv]`` — single raw-argmax estimate.
    """
    if alphas is None:
        alphas = _cfg.ALPHAS

    results_dir = ensure_npz_dir()
    all_results = {}

    for alpha in alphas:
        print(f"\n[Mean] alpha={alpha:.3f} ...")
        res = compute_mean_method(
            alpha,
            beta_min=beta_min, beta_max=beta_max, stride=stride,
            file_template=file_template,
        )
        all_results[alpha] = res

        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print(f"  beta_c = {res['beta_c']:.2f}  FSS={fss:.2f}  "
              f"(n_features={res['n_features']}, n_samples={res['n_samples']})")

    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]   = r["u_beta"]
        store_snapshots(save_dict, key, r)
        save_dict[f"{key}_x_bar"]    = r["x_bar"]
        save_dict[f"{key}_delta"]    = r["delta"]
        save_dict[f"{key}_beta_mid"] = r["beta_mid"]
        save_dict[f"{key}_beta_c"]   = np.array([r["beta_c"]])

    fname = result_filename(
        "mean",
        make_tag(beta_min, file_template, tag,
                 beta_max=beta_max, stride=stride),
    )
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Mean-based β_c extraction (Arnold 2021, arXiv:2010.04730)"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Named model preset (sets alphas, file-template, FSS)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--file-template", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if args.model is not None:
        _cfg.apply_model(args.model)
    if args.data_dir is not None:
        _cfg.DATA_DIR = args.data_dir

    alphas        = args.alphas        or _cfg.ALPHAS
    file_template = args.file_template or _cfg.FILE_TEMPLATE

    run_all(alphas=alphas, beta_min=args.beta_min, beta_max=args.beta_max,
            stride=args.stride, file_template=file_template, tag=args.tag)


if __name__ == "__main__":
    main()
