"""
PCA-based critical-point (β_c / α_c) extraction.

Two estimators, both built from the first two principal components of the
centred data:

  1. Susceptibility peak — Var(P_k)(x) is the PCA susceptibility and peaks at
     the transition.  (So when <|P_k|> is V-shaped, x_c is the *variance* peak,
     not the <|P_k|> minimum.)  The smoothed argmax is the noise-robust value.

  2. KMeans (k=2) on the (P1, P2) cloud — the cluster with the higher mean
     control parameter is the ordered phase; its ordered-fraction f(x) is fit
     with a logistic sigmoid whose midpoint is x_c.

A β-sweep (vary β at fixed α) and an α-sweep (vary α at fixed β) share all of
the above; the α-sweep only flips the convention so that *high α* is ordered.

CLI
---
    python3.11 -m ssh_ml_phase.pca_method                       # all alphas
    python3.11 -m ssh_ml_phase.pca_method --alphas 0.240        # one alpha
    python3.11 -m ssh_ml_phase.pca_method --beta-min 8.0        # crop low β
    python3.11 -m ssh_ml_phase.pca_method --alpha-sweep --betas 6.0
"""
import argparse
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from . import config as _cfg
from .data import (load_beta_sweep, load_alpha_sweep, ensure_npz_dir, make_tag,
                   result_filename, lattice_snapshots, store_snapshots)

_SMOOTH   = 2      # Gaussian sigma for the *_sm smoothed curves
_RNG_SEED = 42     # scatter-subsample seed (reproducible plots)


# ── Sigmoid (also imported by the notebook) ──────────────────────────────────

def sigmoid_fit(x, x_c, w):
    """Logistic sigmoid f = 1 / (1 + exp(-(x - x_c)/w))."""
    return 1.0 / (1.0 + np.exp(-(x - x_c) / w))


def sigmoid_deriv(x, x_c, w):
    """df/dx of the logistic sigmoid: f(1-f)/w, peaking at x_c (height 1/4w)."""
    s = sigmoid_fit(x, x_c, w)
    return s * (1.0 - s) / w


# ── Clustering recipes for the post-PCA score cloud ───────────────────────────
# Every recipe returns a hard 0/1 label per sample (the downstream code assumes
# exactly two clusters).  `dims` = how many leading PCs to cluster on
# (None → the recipe's natural default: 1 for arcsinh, 2 for the rest).
#   raw_2d     KMeans on the raw PCs.  Default; good when the PCs have
#              comparable variance and a roughly Gaussian shape.
#   spherical  KMeans on P/|P| — clusters by direction only.  Use when |P| has a
#              long tail but its angle is stable across the transition.
#   arcsinh    KMeans on arcsinh(P) (a smooth signed log) — linearises an
#              exponential PC ramp so the split sits near x_c, not in the tail.
#   log        KMeans on signed log, sign(P)·log1p(|P|) — like arcsinh but with a
#              sharper knee near 0.  (Plain log(P) is undefined: PC scores are signed.)
#   gmm        2-component Gaussian mixture (full covariance) — fits tilted,
#              unequal-variance ellipses instead of KMeans' round Euclidean blobs.
VALID_PCA_CLUSTER_METHODS = ("raw_2d", "spherical", "arcsinh", "log", "gmm")


def _cluster_pca_scores(P, method="raw_2d", dims=None):
    """Hard 0/1 cluster labels for the PC scores under the chosen recipe.

    dims : int or None
        Number of leading PCs to cluster on.  None → 1 for "arcsinh", else 2.
        Clamped to the PCs available, so over-asking is safe.
    """
    n = dims if dims is not None else (1 if method == "arcsinh" else 2)
    X = P[:, :n]
    km = KMeans(n_clusters=2, random_state=10, n_init=25, init='random')
    if method == "raw_2d":
        return km.fit_predict(X)
    if method == "spherical":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return km.fit_predict(X / np.maximum(norms, 1e-12))
    if method == "arcsinh":
        return km.fit_predict(np.arcsinh(X))
    if method == "log":
        return km.fit_predict(np.sign(X) * np.log1p(np.abs(X)))
    if method == "gmm":
        return GaussianMixture(n_components=2, covariance_type="spherical",
                               n_init=10, random_state=42).fit_predict(X)
    raise ValueError(f"unknown cluster_method {method!r}; "
                     f"valid: {VALID_PCA_CLUSTER_METHODS}")


# ── Shared building blocks (used by both the β- and α-sweep) ──────────────────

def _pc_stats(P, group, u):
    """Per-value mean / |mean| / variance of PC1 and PC2 (plus smoothed
    versions), where ``group[i]`` is the control parameter of sample i and
    ``u`` its sorted unique values."""
    col = lambda k, fn: np.array([fn(P[group == v, k]) for v in u])
    mean_abs_P1 = col(0, lambda x: np.abs(x).mean())
    var_P1      = col(0, np.var)
    var_P2      = col(1, np.var)
    return {
        "mean_P1":        col(0, np.mean),
        "mean_P2":        col(1, np.mean),
        "mean_abs_P1":    mean_abs_P1,
        "mean_abs_P1_sm": gaussian_filter1d(mean_abs_P1, _SMOOTH),
        "var_P1":         var_P1,
        "var_P1_sm":      gaussian_filter1d(var_P1, _SMOOTH),
        "mean_abs_P2":    col(1, lambda x: np.abs(x).mean()),
        "var_P2":         var_P2,
        "var_P2_sm":      gaussian_filter1d(var_P2, _SMOOTH),
    }


def _susceptibility_estimators(u, s):
    """x_c from the variance peaks (raw and smoothed argmax) and from the
    <|P1|> inflection (argmax of its derivative).  ``s`` is a ``_pc_stats``
    dict; also returns (x_mid, dy/dx) curves for plotting."""
    mid = 0.5 * (u[:-1] + u[1:])
    d   = lambda y: np.diff(y) / np.diff(u)
    return {
        "argmax_P1":           float(u[np.argmax(s["var_P1"])]),
        "argmax_P2":           float(u[np.argmax(s["var_P2"])]),
        "sm_argmax_P1":        float(u[np.argmax(s["var_P1_sm"])]),
        "sm_argmax_P2":        float(u[np.argmax(s["var_P2_sm"])]),
        "mean_P1_deriv":       float(mid[np.argmax(d(s["mean_abs_P1_sm"]))]),
        "var1_deriv_curve":    (mid, d(s["var_P1_sm"])),
        "var2_deriv_curve":    (mid, d(s["var_P2_sm"])),
        "mean_P1_deriv_curve": (mid, d(s["mean_abs_P1_sm"])),
    }


def _kmeans_estimators(P, group, u, cluster_method, x0, w0, w_lo, w_hi,
                       cluster_dims=None):
    """KMeans ordered-fraction f(x), its sigmoid-midpoint x_c (+1σ) and its
    derivative-peak x_c.  ``x0``/``w0`` seed the sigmoid fit; ``w_lo``/``w_hi``
    bound its width; ``cluster_dims`` sets how many PCs the recipe clusters on."""
    labels  = _cluster_pca_scores(P, method=cluster_method, dims=cluster_dims)
    ordered = int(np.mean(group[labels == 1]) > np.mean(group[labels == 0]))
    f = np.array([np.mean(labels[group == v] == ordered) for v in u])
    try:
        popt, pcov = curve_fit(sigmoid_fit, u, f, p0=[x0, w0],
                               bounds=([u[2], w_lo], [u[-3], w_hi]), maxfev=10000)
        x_c, x_c_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        x_c, x_c_err, popt = np.nan, np.nan, None
    mid = 0.5 * (u[:-1] + u[1:])
    d_f = np.diff(f) / np.diff(u)
    return {
        "km_labels":      labels,
        "f":              f,
        "km_sig":         x_c,
        "km_sig_err":     x_c_err,
        "km_sig_popt":    popt,
        "km_deriv":       float(mid[np.argmax(d_f)]),
        "km_deriv_curve": (mid, d_f),
    }


def _zscore_crossings(u, mean_P1, mean_P2, s1, s2):
    """x where the z-scored per-value means of PC1 and PC2 cross — the linearly
    interpolated sign-changes of (mean_P1/σ1 − mean_P2/σ2)."""
    m1   = mean_P1 / s1 if s1 > 0 else mean_P1
    m2   = mean_P2 / s2 if s2 > 0 else mean_P2
    diff = m1 - m2
    out  = []
    for i in np.where(np.diff(np.sign(diff)))[0]:
        d0, d1 = diff[i], diff[i + 1]
        if d1 != d0:
            out.append(float(u[i] - d0 * (u[i + 1] - u[i]) / (d1 - d0)))
    return np.array(out, dtype=float)


def _save_scatter(d, key, P, group, group_name, labels, n=5000):
    """Subsample the (P1, P2, P3) cloud and labels for the embedding scatter
    plots (P3 feeds the 3D embedding view)."""
    idx = np.random.default_rng(_RNG_SEED).choice(len(P), min(n, len(P)),
                                                  replace=False)
    d[f"{key}_P1_sample"]           = P[idx, 0]
    d[f"{key}_P2_sample"]           = P[idx, 1]
    d[f"{key}_P3_sample"]           = P[idx, 2]
    d[f"{key}_{group_name}_sample"] = group[idx]
    d[f"{key}_km_labels_sample"]    = labels[idx]


# ── β-sweep (vary β at fixed α) ───────────────────────────────────────────────

def run_pca(alpha, n_components=10, beta_min=None, beta_max=None,
            stride=None, file_template=None):
    """PCA of the β-sweep at one α; returns the scores, per-β PC statistics,
    the PC1/PC2 loadings and the per-β input-lattice snapshots."""
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    pca = PCA(n_components=n_components)
    P = pca.fit_transform(X - X.mean(axis=0))
    u_beta = np.unique(betas)
    inlat_betas, inlat_grids = lattice_snapshots(X, betas)
    return {
        "P": P, "betas": betas, "u_beta": u_beta,
        "inlat_betas": inlat_betas, "inlat_grids": inlat_grids,
        "explained_var": pca.explained_variance_ratio_,
        "pc1_loading": pca.components_[0], "pc2_loading": pca.components_[1],
        **_pc_stats(P, betas, u_beta),
    }


def extract_beta_c(alpha, pca_result=None, beta_min=None, beta_max=None,
                   stride=None, file_template=None, cluster_method="raw_2d",
                   cluster_dims=None):
    """β_c from the susceptibility peaks and the KMeans ordered-fraction
    sigmoid.  ``best`` is the smoothed Var(P1) argmax."""
    pr = pca_result or run_pca(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    u      = pr["u_beta"]
    fss_bc = _cfg.FSS_DICT.get(alpha, np.mean(u))

    sus = _susceptibility_estimators(u, pr)
    km  = _kmeans_estimators(pr["P"], pr["betas"], u, cluster_method,
                             x0=fss_bc, w0=2.0, w_lo=0.1, w_hi=20.0,
                             cluster_dims=cluster_dims)
    return {
        **{k: sus[k] for k in ("argmax_P1", "argmax_P2", "sm_argmax_P1",
                               "sm_argmax_P2", "var1_deriv_curve",
                               "var2_deriv_curve", "mean_P1_deriv_curve")},
        "mean_P1_deriv_bc": sus["mean_P1_deriv"],
        "best":             sus["sm_argmax_P1"],
        "km_sig": km["km_sig"], "km_sig_err": km["km_sig_err"],
        "km_sig_popt": km["km_sig_popt"], "km_deriv": km["km_deriv"],
        "km_deriv_curve": km["km_deriv_curve"], "km_labels": km["km_labels"],
        "f_kvbs_km": km["f"],
        "crossings_P1P2": _zscore_crossings(
            u, pr["mean_P1"], pr["mean_P2"],
            pr["P"][:, 0].std(), pr["P"][:, 1].std()),
        "pca_result": pr,
    }


def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None, cluster_method="raw_2d",
            cluster_dims=None):
    """Run the β-sweep PCA for every α and save results/npz/pca_results[...].npz.

    beta_c array layout:
        [argmax_P1, argmax_P2, km_sig, km_sig_err, km_deriv, best,
         mean_P1_deriv_bc]
    """
    alphas = _cfg.ALPHAS if alphas is None else alphas
    results_dir = ensure_npz_dir()
    results, save = {}, {}

    for alpha in alphas:
        print(f"[PCA] alpha={alpha:.3f}  cluster_method={cluster_method}"
              f"  dims={cluster_dims} ...")
        r = extract_beta_c(alpha, beta_min=beta_min, beta_max=beta_max,
                           stride=stride, file_template=file_template,
                           cluster_method=cluster_method,
                           cluster_dims=cluster_dims)
        results[alpha] = r
        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print(f"  sm_argmax_P1={r['sm_argmax_P1']:.1f}  "
              f"mean_P1_deriv={r['mean_P1_deriv_bc']:.1f}  "
              f"km_sig={r['km_sig']:.1f}±{r['km_sig_err']:.2f}  "
              f"km_deriv={r['km_deriv']:.1f}  FSS={fss:.1f}")

        pr, key = r["pca_result"], f"a{alpha:.4f}"
        save[f"{key}_u_beta"]      = pr["u_beta"]
        save[f"{key}_mean_P2"]     = pr["mean_P2"]
        save[f"{key}_pc1_loading"] = pr["pc1_loading"]
        save[f"{key}_pc2_loading"] = pr["pc2_loading"]
        for k in ("mean_P1", "mean_abs_P1", "mean_abs_P1_sm", "var_P1",
                  "var_P1_sm", "mean_abs_P2", "var_P2", "var_P2_sm",
                  "explained_var"):
            save[f"{key}_{k}"] = pr[k]
        store_snapshots(save, key, pr)
        save[f"{key}_crossings_P1P2"] = r["crossings_P1P2"]
        save[f"{key}_beta_c"] = np.array([
            r["argmax_P1"], r["argmax_P2"], r["km_sig"], r["km_sig_err"],
            r["km_deriv"], r["best"], r["mean_P1_deriv_bc"]])
        save[f"{key}_sm_argmax_P1"]     = r["sm_argmax_P1"]
        save[f"{key}_sm_argmax_P2"]     = r["sm_argmax_P2"]
        save[f"{key}_mean_P1_deriv_bc"] = r["mean_P1_deriv_bc"]
        save[f"{key}_var1_deriv_beta"],     save[f"{key}_var1_deriv_vals"]     = r["var1_deriv_curve"]
        save[f"{key}_var2_deriv_beta"],     save[f"{key}_var2_deriv_vals"]     = r["var2_deriv_curve"]
        save[f"{key}_mean_P1_deriv_beta"],  save[f"{key}_mean_P1_deriv_vals"]  = r["mean_P1_deriv_curve"]
        save[f"{key}_f_kvbs_km"]         = r["f_kvbs_km"]
        save[f"{key}_km_deriv_beta"],    save[f"{key}_km_deriv_vals"]    = r["km_deriv_curve"]
        if r["km_sig_popt"] is not None:
            save[f"{key}_km_sig_popt"] = np.array(r["km_sig_popt"])
        _save_scatter(save, key, pr["P"], pr["betas"], "betas", r["km_labels"])

    fname = result_filename("pca", make_tag(beta_min, file_template, tag,
                                            beta_max=beta_max, stride=stride))
    np.savez(f"{results_dir}/{fname}", **save)
    print(f"\nResults saved to {results_dir}/{fname}")
    return results


# ── α-sweep (vary α at fixed β; high α = ordered/CDW phase) ────────────────────

def run_pca_alpha_sweep(beta, file_template, n_components=10,
                        alpha_min=None, alpha_max=None):
    """PCA of the α-sweep at one β.  Same per-value statistics as run_pca, keyed
    on α (no PC loadings or input-lattice snapshots are kept for α-sweeps)."""
    X, alphas = load_alpha_sweep(beta, file_template=file_template,
                                 alpha_min=alpha_min, alpha_max=alpha_max)
    pca = PCA(n_components=n_components)
    P = pca.fit_transform(X - X.mean(axis=0))
    u_alpha = np.unique(alphas)
    return {
        "P": P, "alphas": alphas, "u_alpha": u_alpha,
        "explained_var": pca.explained_variance_ratio_,
        **_pc_stats(P, alphas, u_alpha),
    }


def extract_alpha_c(beta, file_template, pca_result=None,
                    alpha_min=None, alpha_max=None, fss_alpha_c=None,
                    cluster_method="raw_2d", cluster_dims=None):
    """α_c analogue of extract_beta_c (high α = ordered/CDW phase).  ``fss_alpha_c``
    seeds the sigmoid fit and defaults to the midpoint of the α grid."""
    pr = pca_result or run_pca_alpha_sweep(beta, file_template,
                                           alpha_min=alpha_min, alpha_max=alpha_max)
    u = pr["u_alpha"]
    if fss_alpha_c is None:
        fss_alpha_c = float(np.mean(u))

    sus = _susceptibility_estimators(u, pr)
    km  = _kmeans_estimators(pr["P"], pr["alphas"], u, cluster_method,
                             x0=fss_alpha_c, w0=0.3, w_lo=0.01, w_hi=5.0,
                             cluster_dims=cluster_dims)
    return {
        **{k: sus[k] for k in ("argmax_P1", "argmax_P2", "sm_argmax_P1",
                               "sm_argmax_P2", "var1_deriv_curve",
                               "var2_deriv_curve", "mean_P1_deriv_curve")},
        "mean_P1_deriv_ac": sus["mean_P1_deriv"],
        "best":             sus["sm_argmax_P1"],
        "km_sig": km["km_sig"], "km_sig_err": km["km_sig_err"],
        "km_sig_popt": km["km_sig_popt"], "km_deriv": km["km_deriv"],
        "km_deriv_curve": km["km_deriv_curve"], "km_labels": km["km_labels"],
        "f_cdw_km": km["f"],
        "pca_result": pr,
    }


def run_all_alpha_sweep(betas, file_template, alpha_min=None, alpha_max=None,
                        fss_alpha_c=None, tag=None, cluster_method="raw_2d",
                        cluster_dims=None):
    """Run the α-sweep PCA for each fixed-β file and save
    results/npz/pca_results_alpha_sweep[_tag].npz.  The alpha_c array mirrors
    run_all's beta_c layout (last entry is mean_P1_deriv_ac)."""
    results_dir = ensure_npz_dir()
    results, save = {}, {}

    for beta in betas:
        print(f"[PCA alpha-sweep] beta={beta:.2f}  "
              f"cluster_method={cluster_method}  dims={cluster_dims} ...")
        r = extract_alpha_c(beta, file_template, alpha_min=alpha_min,
                            alpha_max=alpha_max, fss_alpha_c=fss_alpha_c,
                            cluster_method=cluster_method,
                            cluster_dims=cluster_dims)
        results[beta] = r
        fss_str = f"{fss_alpha_c:.3f}" if fss_alpha_c is not None else "None"
        print(f"  sm_argmax_P1={r['sm_argmax_P1']:.3f}  "
              f"km_sig={r['km_sig']:.3f}±{r['km_sig_err']:.4f}  "
              f"km_deriv={r['km_deriv']:.3f}  FSS_alpha_c={fss_str}")

        pr, key = r["pca_result"], f"b{beta:.2f}"
        save[f"{key}_u_alpha"] = pr["u_alpha"]
        for k in ("mean_P1", "mean_abs_P1", "mean_abs_P1_sm", "var_P1",
                  "var_P1_sm", "mean_abs_P2", "var_P2", "var_P2_sm",
                  "explained_var"):
            save[f"{key}_{k}"] = pr[k]
        save[f"{key}_alpha_c"] = np.array([
            r["argmax_P1"], r["argmax_P2"], r["km_sig"], r["km_sig_err"],
            r["km_deriv"], r["best"], r["mean_P1_deriv_ac"]])
        save[f"{key}_f_cdw_km"] = r["f_cdw_km"]
        save[f"{key}_km_deriv_alpha"], save[f"{key}_km_deriv_vals"] = r["km_deriv_curve"]
        if r["km_sig_popt"] is not None:
            save[f"{key}_km_sig_popt"] = np.array(r["km_sig_popt"])
        _save_scatter(save, key, pr["P"], pr["alphas"], "alphas", r["km_labels"])

    suffix = f"_{tag}" if tag else "_alpha_sweep"
    np.savez(f"{results_dir}/pca_results{suffix}.npz", **save)
    print(f"\nResults saved to {results_dir}/pca_results{suffix}.npz")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA-based beta_c extraction")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Named model preset (sets alphas, file-template, FSS). "
                             "Explicit --alphas / --file-template override the preset.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values to process (default: all)")
    parser.add_argument("--beta-min", type=float, default=None,
                        help="Discard samples with beta < this value")
    parser.add_argument("--beta-max", type=float, default=None,
                        help="Discard samples with beta > this value")
    parser.add_argument("--stride", type=int, default=None,
                        help="Use every N-th row (subsample dense datasets)")
    parser.add_argument("--file-template", type=str, default=None,
                        help="Filename template with {alpha:.4f} placeholder")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing input CSV files")
    parser.add_argument("--tag", type=str, default=None,
                        help="Label appended to output filename (auto-generated if omitted)")
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Alpha-sweep mode: fixed beta, varying alpha. "
                             "Requires --betas and --file-template (or model preset "
                             "with ALPHA_SWEEP_FILE_TEMPLATE).")
    parser.add_argument("--betas", nargs="+", type=float, default=None,
                        help="Fixed beta value(s) for alpha-sweep mode")
    parser.add_argument("--fss-alpha-c", type=float, default=None,
                        help="Expected critical alpha (sigmoid fit initial guess, "
                             "alpha-sweep mode only)")
    args = parser.parse_args()

    if args.model is not None:
        _cfg.apply_model(args.model)
    if args.data_dir is not None:
        _cfg.DATA_DIR = args.data_dir

    if args.alpha_sweep:
        if args.betas is None:
            parser.error("--alpha-sweep requires --betas")
        file_template = (args.file_template
                         or getattr(_cfg, "ALPHA_SWEEP_FILE_TEMPLATE", None))
        if file_template is None:
            parser.error("Alpha-sweep mode needs --file-template or "
                         "ALPHA_SWEEP_FILE_TEMPLATE in model config")
        run_all_alpha_sweep(betas=args.betas, file_template=file_template,
                            fss_alpha_c=args.fss_alpha_c, tag=args.tag)
    else:
        alphas        = args.alphas        or _cfg.ALPHAS
        file_template = args.file_template or _cfg.FILE_TEMPLATE
        run_all(alphas=alphas, beta_min=args.beta_min, beta_max=args.beta_max,
                stride=args.stride, file_template=file_template, tag=args.tag)


if __name__ == "__main__":
    main()
