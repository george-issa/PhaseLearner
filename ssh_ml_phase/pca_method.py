"""
PCA-based beta_c extraction using two methods:

1. **Susceptibility peak** — Var(P1)(beta) and Var(P2)(beta) are the PCA
   susceptibilities.  Bond-bond correlation data gives a V-shaped <|P1|>,
   so the *minimum* of <|P1|> is NOT beta_c.  Instead the *peak of Var(Pk)*
   is used.  Two estimates are provided:
   - Raw argmax: beta_c = u_beta[argmax Var(Pk)]
   - Smooth argmax: beta_c = u_beta[argmax Var(Pk_sm)], which is equivalent
     to finding where d Var(Pk_sm)/d beta crosses zero from above.  More
     noise-robust than the raw argmax.

2. **KMeans clustering** — K-means (k=2) on the 2D PCA space (P1 vs P2).
   Identifies the ordered (KVBS) and disordered (SM) phases by cluster
   membership. Fits a sigmoid to f_KVBS(beta) to extract beta_c (with
   1-sigma error), and also takes the derivative peak of f_KVBS as a
   second estimate.

Usage
-----
    python3.11 -m ssh_ml_phase.pca_method                                              # run all alphas
    python3.11 -m ssh_ml_phase.pca_method --alphas 0.240                               # single alpha
    python3.11 -m ssh_ml_phase.pca_method --beta-min 8.0                               # skip low-beta region
    python3.11 -m ssh_ml_phase.pca_method --file-template "GF_L12_a{alpha:.4f}.csv"    # example observable
    python3.11 -m ssh_ml_phase.pca_method --tag myrun                                  # explicit output label
"""
import argparse
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .config import ALPHAS, FSS_BETA_C, FSS_DICT
from .data import load_beta_sweep, ensure_results_dir, make_tag, result_filename


# ── Fit functions ─────────────────────────────────────────────────────────────

def sigmoid_fit(beta, beta_c, w):
    """Standard logistic sigmoid: f = 1 / (1 + exp(-(beta - beta_c)/w))."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


def sigmoid_deriv(beta, beta_c, w):
    """Analytic derivative of the logistic sigmoid: df/dbeta = f(1-f)/w.
    Bell-shaped, peaks exactly at beta = beta_c with height 1/(4w)."""
    s = sigmoid_fit(beta, beta_c, w)
    return s * (1.0 - s) / w


# Kept for backward compatibility (imported by the notebook)
def cosh_fit(beta, A, beta_c, w, C):
    """Legacy V-shape fit (deprecated)."""
    return A * np.cosh((beta - beta_c) / w) + C


def cosh_fit_deriv(beta, A, beta_c, w):
    """Legacy derivative of cosh_fit (deprecated)."""
    return (A / w) * np.sinh((beta - beta_c) / w)


def tanh_fit(beta, A, beta_c, w, B):
    """Legacy monotone fit: f(beta) = A * tanh((beta - beta_c) / w) + B."""
    return A * np.tanh((beta - beta_c) / w) + B


# ── Core PCA routine ──────────────────────────────────────────────────────────

def run_pca(alpha, n_components=10, beta_min=None, beta_max=None,
            stride=None, file_template=None):
    """
    Run PCA and compute per-beta statistics for the first two principal components.

    Parameters
    ----------
    beta_min, beta_max : float or None
        Beta interval bounds; passed to load_beta_sweep (per-alpha config
        values are used when these are None).
    stride : int or None
        Row stride for subsampling dense datasets (passed to load_beta_sweep).
    file_template : str or None
        Override the default filename template (see data.load_beta_sweep).

    Returns
    -------
    dict with keys:
        'P'               — all principal components, shape (N, n_components)
        'betas'           — beta label for every sample
        'u_beta'          — sorted unique beta values
        'mean_P1'         — <P1>(beta)  signed mean (order-parameter view)
        'mean_abs_P1'     — <|P1|>(beta)
        'mean_abs_P1_sm'  — Gaussian-smoothed <|P1|> (sigma=2)
        'var_P1'          — Var(P1)(beta) [susceptibility — peaks near beta_c]
        'var_P1_sm'       — Gaussian-smoothed Var(P1)
        'mean_abs_P2'     — <|P2|>(beta)
        'var_P2'          — Var(P2)(beta)
        'var_P2_sm'       — Gaussian-smoothed Var(P2)
        'explained_var'   — PCA explained variance ratios
    """
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=n_components)
    P = pca.fit_transform(X_centered)

    u_beta      = np.unique(betas)
    mean_P1     = np.array([np.mean(P[betas == b, 0])         for b in u_beta])
    mean_abs_P1 = np.array([np.mean(np.abs(P[betas == b, 0])) for b in u_beta])
    var_P1      = np.array([np.var(P[betas == b, 0])           for b in u_beta])
    mean_abs_P2 = np.array([np.mean(np.abs(P[betas == b, 1])) for b in u_beta])
    var_P2      = np.array([np.var(P[betas == b, 1])           for b in u_beta])

    mean_abs_P1_sm = gaussian_filter1d(mean_abs_P1, sigma=2)

    return {
        "P":               P,
        "betas":           betas,
        "u_beta":          u_beta,
        "mean_P1":         mean_P1,
        "mean_abs_P1":     mean_abs_P1,
        "mean_abs_P1_sm":  mean_abs_P1_sm,
        "var_P1":          var_P1,
        "var_P1_sm":       gaussian_filter1d(var_P1, sigma=2),
        "mean_abs_P2":     mean_abs_P2,
        "var_P2":          var_P2,
        "var_P2_sm":       gaussian_filter1d(var_P2, sigma=2),
        "explained_var":   pca.explained_variance_ratio_,
    }


# ── Beta_c extraction ─────────────────────────────────────────────────────────

def extract_beta_c(alpha, pca_result=None, beta_min=None, beta_max=None,
                   stride=None, file_template=None):
    """
    Extract beta_c using PCA susceptibility and clustering methods.

    For bond-bond correlation data, <|P1|>(beta) is V-shaped and its minimum
    is NOT beta_c.  Instead we use the *variance* Var(P_k)(beta), which is the
    PCA susceptibility and peaks at beta_c.  beta_c is read off as the argmax
    of the raw variance arrays.

    Parameters
    ----------
    beta_min : float or None
        Passed to run_pca if pca_result is not already provided.
    file_template : str or None
        Passed to run_pca if pca_result is not already provided.

    Returns dict with keys:
        'argmax_P1'       — beta_c = u_beta[argmax Var(P1)]  (raw)
        'argmax_P2'       — beta_c = u_beta[argmax Var(P2)]  (raw)
        'sm_argmax_P1'    — beta_c = u_beta[argmax Var(P1_sm)], equiv. to the
                            zero-crossing of d Var(P1_sm)/d beta from above
        'sm_argmax_P2'    — same for P2
        'km_sig'              — beta_c from KMeans sigmoid fit on 2D PCA space
        'km_sig_err'          — 1-sigma error on km_sig from covariance matrix
        'km_deriv'            — beta_c from point-wise derivative peak of f_KVBS
        'mean_P1_deriv_bc'    — beta_c from argmax of d<|P1|>/dbeta.
                                For observables where <|P1|> is a monotone
                                order parameter (e.g. G(k)), this is the
                                inflection-point estimator, complementary to
                                the variance susceptibility peak.
        'best'                — sm_argmax_P1
        'var1_deriv_curve'    — (beta_mid, d(Var_P1_sm)/dbeta) for viz
        'var2_deriv_curve'    — (beta_mid, d(Var_P2_sm)/dbeta) for viz
        'mean_P1_deriv_curve' — (beta_mid, d(<|P1|>_sm)/dbeta) for viz
        'km_deriv_curve'      — (beta_mid, d(f_KVBS)/dbeta) for viz
        plus 'km_labels', 'f_kvbs_km', 'km_sig_popt', 'pca_result'
    """
    if pca_result is None:
        pca_result = run_pca(alpha, beta_min=beta_min, beta_max=beta_max,
                             stride=stride, file_template=file_template)

    u_beta          = pca_result["u_beta"]
    var_P1          = pca_result["var_P1"]
    var_P1_sm       = pca_result["var_P1_sm"]
    var_P2          = pca_result["var_P2"]
    var_P2_sm       = pca_result["var_P2_sm"]
    mean_abs_P1_sm  = pca_result["mean_abs_P1_sm"]
    P               = pca_result["P"]
    betas_all       = pca_result["betas"]
    fss_bc          = FSS_DICT.get(alpha, np.mean(u_beta))

    # ── Method 1: argmax of Var(P1) ──────────────────────────────────────────
    bc_argmax_P1 = float(u_beta[np.argmax(var_P1)])

    # ── Method 2: argmax of Var(P2) ──────────────────────────────────────────
    bc_argmax_P2 = float(u_beta[np.argmax(var_P2)])

    # ── Smooth argmax / zero-crossing of d(Var_sm)/d beta ────────────────────
    # argmax(Var_Pk_sm) is the noise-robust susceptibility peak estimate.
    # It is equivalent to finding where d(Var_Pk_sm)/dbeta crosses zero from above.
    bc_sm_argmax_P1 = float(u_beta[np.argmax(var_P1_sm)])
    bc_sm_argmax_P2 = float(u_beta[np.argmax(var_P2_sm)])

    # Derivative curves of smoothed Var — saved for visualization only.
    beta_mid   = 0.5 * (u_beta[:-1] + u_beta[1:])
    d_var_sm1  = np.diff(var_P1_sm) / np.diff(u_beta)
    d_var_sm2  = np.diff(var_P2_sm) / np.diff(u_beta)

    # ── Order-parameter derivative estimator ──────────────────────────────────
    # d<|P1|>/dbeta peaks at the inflection point of the monotone order
    # parameter — a complementary beta_c estimator for observables like G(k)
    # where Var(P1) does not peak but <|P1|> rises monotonically.
    d_mean_abs_P1  = np.diff(mean_abs_P1_sm) / np.diff(u_beta)
    bc_mean_P1_deriv = float(beta_mid[np.argmax(d_mean_abs_P1)])

    # ── KMeans clustering on 2D PCA space ────────────────────────────────────
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = km.fit_predict(P[:, :2])

    # Cluster with higher mean beta = ordered (KVBS) phase
    mean_beta_0 = np.mean(betas_all[km_labels == 0])
    mean_beta_1 = np.mean(betas_all[km_labels == 1])
    kvbs_label_km = 1 if mean_beta_1 > mean_beta_0 else 0

    f_kvbs_km = np.array(
        [np.mean(km_labels[betas_all == b] == kvbs_label_km) for b in u_beta]
    )

    try:
        popt_km, pcov_km = curve_fit(
            sigmoid_fit, u_beta, f_kvbs_km,
            p0=[fss_bc, 2.0],
            bounds=([u_beta[2], 0.1], [u_beta[-3], 20.0]),
            maxfev=10000,
        )
        bc_km     = float(popt_km[0])
        bc_km_err = float(np.sqrt(pcov_km[0, 0]))
    except Exception:
        bc_km     = np.nan
        bc_km_err = np.nan
        popt_km   = None

    d_fkm       = np.diff(f_kvbs_km) / np.diff(u_beta)
    bc_km_deriv = float(beta_mid[np.argmax(d_fkm)])

    return {
        "argmax_P1":          bc_argmax_P1,
        "argmax_P2":          bc_argmax_P2,
        "sm_argmax_P1":       bc_sm_argmax_P1,
        "sm_argmax_P2":       bc_sm_argmax_P2,
        "km_sig":             bc_km,
        "km_sig_err":         bc_km_err,
        "km_deriv":           bc_km_deriv,
        "mean_P1_deriv_bc":   bc_mean_P1_deriv,
        "best":               bc_sm_argmax_P1,
        "var1_deriv_curve":   (beta_mid, d_var_sm1),
        "var2_deriv_curve":   (beta_mid, d_var_sm2),
        "mean_P1_deriv_curve":(beta_mid, d_mean_abs_P1),
        "km_deriv_curve":     (beta_mid, d_fkm),
        "km_labels":          km_labels,
        "f_kvbs_km":          f_kvbs_km,
        "km_sig_popt":        popt_km,
        "pca_result":         pca_result,
    }


def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None):
    """
    Run PCA analysis for all alphas. Saves results to results/pca_results.npz.

    beta_c array layout:
        [argmax_P1, argmax_P2, km_sig, km_sig_err, km_deriv, best,
         mean_P1_deriv_bc]
    """
    if alphas is None:
        alphas = ALPHAS

    results_dir = ensure_results_dir()
    all_results = {}

    for i, alpha in enumerate(alphas):
        print(f"[PCA] alpha={alpha:.3f} ...")
        res = extract_beta_c(alpha, beta_min=beta_min, beta_max=beta_max,
                             stride=stride, file_template=file_template)
        all_results[alpha] = res

        fss = FSS_DICT.get(alpha, np.nan)
        print(
            f"  sm_argmax_P1={res['sm_argmax_P1']:.1f}  "
            f"mean_P1_deriv={res['mean_P1_deriv_bc']:.1f}  "
            f"km_sig={res['km_sig']:.1f}±{res['km_sig_err']:.2f}  "
            f"km_deriv={res['km_deriv']:.1f}  FSS={fss:.1f}"
        )

    # Save
    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        pr = r["pca_result"]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]           = pr["u_beta"]
        save_dict[f"{key}_mean_P1"]          = pr["mean_P1"]
        save_dict[f"{key}_mean_abs_P1"]      = pr["mean_abs_P1"]
        save_dict[f"{key}_mean_abs_P1_sm"]   = pr["mean_abs_P1_sm"]
        save_dict[f"{key}_var_P1"]           = pr["var_P1"]
        save_dict[f"{key}_var_P1_sm"]        = pr["var_P1_sm"]
        save_dict[f"{key}_mean_abs_P2"]      = pr["mean_abs_P2"]
        save_dict[f"{key}_var_P2"]           = pr["var_P2"]
        save_dict[f"{key}_var_P2_sm"]        = pr["var_P2_sm"]
        save_dict[f"{key}_explained_var"]    = pr["explained_var"]
        # beta_c: [argmax_P1, argmax_P2, km_sig, km_sig_err, km_deriv, best,
        #          mean_P1_deriv_bc]
        save_dict[f"{key}_beta_c"] = np.array([
            r["argmax_P1"], r["argmax_P2"],
            r["km_sig"], r["km_sig_err"], r["km_deriv"], r["best"],
            r["mean_P1_deriv_bc"],
        ])
        save_dict[f"{key}_sm_argmax_P1"]        = r["sm_argmax_P1"]
        save_dict[f"{key}_sm_argmax_P2"]        = r["sm_argmax_P2"]
        save_dict[f"{key}_mean_P1_deriv_bc"]    = r["mean_P1_deriv_bc"]
        # Var derivative curves
        save_dict[f"{key}_var1_deriv_beta"]     = r["var1_deriv_curve"][0]
        save_dict[f"{key}_var1_deriv_vals"]     = r["var1_deriv_curve"][1]
        save_dict[f"{key}_var2_deriv_beta"]     = r["var2_deriv_curve"][0]
        save_dict[f"{key}_var2_deriv_vals"]     = r["var2_deriv_curve"][1]
        # <|P1|> derivative curve (order-parameter inflection estimator)
        save_dict[f"{key}_mean_P1_deriv_beta"]  = r["mean_P1_deriv_curve"][0]
        save_dict[f"{key}_mean_P1_deriv_vals"]  = r["mean_P1_deriv_curve"][1]
        # KMeans data
        save_dict[f"{key}_f_kvbs_km"]       = r["f_kvbs_km"]
        save_dict[f"{key}_km_deriv_beta"]   = r["km_deriv_curve"][0]
        save_dict[f"{key}_km_deriv_vals"]   = r["km_deriv_curve"][1]
        if r["km_sig_popt"] is not None:
            save_dict[f"{key}_km_sig_popt"] = np.array(r["km_sig_popt"])
        # Scatter data (subsample for visualization)
        P = pr["P"]
        betas = pr["betas"]
        idx = np.random.default_rng(42).choice(len(P), min(5000, len(P)), replace=False)
        save_dict[f"{key}_P1_sample"]        = P[idx, 0]
        save_dict[f"{key}_P2_sample"]        = P[idx, 1]
        save_dict[f"{key}_betas_sample"]     = betas[idx]
        save_dict[f"{key}_km_labels_sample"] = r["km_labels"][idx]

    fname = result_filename("pca", make_tag(beta_min, file_template, tag,
                                           beta_max=beta_max, stride=stride))
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="PCA-based beta_c extraction")
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
    parser.add_argument("--tag", type=str, default=None,
                        help="Label appended to output filename (auto-generated if omitted)")
    args = parser.parse_args()
    run_all(alphas=args.alphas, beta_min=args.beta_min, beta_max=args.beta_max,
            stride=args.stride, file_template=args.file_template, tag=args.tag)


if __name__ == "__main__":
    main()
