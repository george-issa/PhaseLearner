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


# ── Clustering recipes for the post-PCA step ─────────────────────────────────
# The PCA itself is always linear (no rescaling, no log).  These recipes only
# choose how KMeans interprets the PC scores for cluster assignment.
#
#   "raw_2d"     — KMeans(k=2) on raw (P1, P2).  Default; correct when both PCs
#                  have well-behaved (≈ Gaussian, comparable variance)
#                  distributions.  Fails on heavy-tailed observables (e.g. SC
#                  susceptibility) because KMeans is dominated by PC1's tail.
#   "spherical"  — L2-normalise each (P1, P2) → KMeans on the unit circle.
#                  Drops magnitude entirely; clusters by direction (angle).
#                  The right choice when |P| has a long tail but
#                  arg(P1, P2) is stable across the transition.
#   "arcsinh"    — 1-D KMeans on arcsinh(P1).  arcsinh is a smooth signed log:
#                  linear near 0, ~ ln(2|x|) in the tails.  Linearises an
#                  exponential PC1 ramp so the centroid-pair midpoint sits
#                  near β_c instead of in the tail.

VALID_PCA_CLUSTER_METHODS = ("raw_2d", "spherical", "arcsinh")


def _cluster_pca_scores(P, method="raw_2d"):
    """Return KMeans(k=2) cluster labels for the given PC scores under the
    requested clustering recipe (see module-level table)."""
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    if method == "raw_2d":
        return km.fit_predict(P[:, :2])
    if method == "spherical":
        norms = np.linalg.norm(P[:, :2], axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return km.fit_predict(P[:, :2] / norms)
    if method == "arcsinh":
        return km.fit_predict(np.arcsinh(P[:, :1]))
    raise ValueError(
        f"unknown PCA cluster_method {method!r}; "
        f"valid: {VALID_PCA_CLUSTER_METHODS}"
    )

from . import config as _cfg
from .data import load_beta_sweep, load_alpha_sweep, ensure_npz_dir, make_tag, result_filename


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
        'mean_P2'         — <P2>(beta)  signed mean
        'mean_abs_P1'     — <|P1|>(beta)
        'mean_abs_P1_sm'  — Gaussian-smoothed <|P1|> (sigma=2)
        'var_P1'          — Var(P1)(beta) [susceptibility — peaks near beta_c]
        'var_P1_sm'       — Gaussian-smoothed Var(P1)
        'mean_abs_P2'     — <|P2|>(beta)
        'var_P2'          — Var(P2)(beta)
        'var_P2_sm'       — Gaussian-smoothed Var(P2)
        'explained_var'   — PCA explained variance ratios
        'pc1_loading'     — PCA component vector for PC1, shape (n_features,)
    """
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=n_components)
    P = pca.fit_transform(X_centered)

    u_beta      = np.unique(betas)
    mean_P1     = np.array([np.mean(P[betas == b, 0])         for b in u_beta])
    mean_P2     = np.array([np.mean(P[betas == b, 1])         for b in u_beta])
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
        "mean_P2":         mean_P2,
        "mean_abs_P1":     mean_abs_P1,
        "mean_abs_P1_sm":  mean_abs_P1_sm,
        "var_P1":          var_P1,
        "var_P1_sm":       gaussian_filter1d(var_P1, sigma=2),
        "mean_abs_P2":     mean_abs_P2,
        "var_P2":          var_P2,
        "var_P2_sm":       gaussian_filter1d(var_P2, sigma=2),
        "explained_var":   pca.explained_variance_ratio_,
        "pc1_loading":     pca.components_[0],
    }


# ── Beta_c extraction ─────────────────────────────────────────────────────────

def extract_beta_c(alpha, pca_result=None, beta_min=None, beta_max=None,
                   stride=None, file_template=None, cluster_method="raw_2d"):
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
    fss_bc          = _cfg.FSS_DICT.get(alpha, np.mean(u_beta))

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

    # ── KMeans clustering on PC scores (recipe selectable) ───────────────────
    # PCA itself is always linear; this step chooses how KMeans interprets
    # the (P1, P2) cloud.  See `_cluster_pca_scores` for the recipe table.
    km_labels = _cluster_pca_scores(P, method=cluster_method)

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

    # ── P1 / P2 z-scored mean crossings ───────────────────────────────────────
    # Each PC's per-β mean is divided by its global std (≈ √λ_k); the
    # difference's sign-changes are linearly interpolated to give β values
    # where the two PCs carry equal signed signal in σ-units.
    s1 = float(P[:, 0].std())
    s2 = float(P[:, 1].std())
    m1z = pca_result["mean_P1"] / s1 if s1 > 0 else pca_result["mean_P1"]
    m2z = pca_result["mean_P2"] / s2 if s2 > 0 else pca_result["mean_P2"]
    diff = m1z - m2z
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    crossings = []
    for i in sign_changes:
        b0, b1 = u_beta[i], u_beta[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        if d1 == d0:
            continue
        crossings.append(float(b0 - d0 * (b1 - b0) / (d1 - d0)))
    crossings_P1P2 = np.array(crossings, dtype=float)

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
        "crossings_P1P2":     crossings_P1P2,
        "pca_result":         pca_result,
    }


def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None, cluster_method="raw_2d"):
    """
    Run PCA analysis for all alphas. Saves results to results/npz/pca_results.npz.

    beta_c array layout:
        [argmax_P1, argmax_P2, km_sig, km_sig_err, km_deriv, best,
         mean_P1_deriv_bc]

    `cluster_method` selects how the PC scores are clustered after PCA;
    see :data:`VALID_PCA_CLUSTER_METHODS` and :func:`_cluster_pca_scores`.
    """
    if alphas is None:
        alphas = _cfg.ALPHAS

    results_dir = ensure_npz_dir()
    all_results = {}

    for i, alpha in enumerate(alphas):
        print(f"[PCA] alpha={alpha:.3f}  cluster_method={cluster_method} ...")
        res = extract_beta_c(alpha, beta_min=beta_min, beta_max=beta_max,
                             stride=stride, file_template=file_template,
                             cluster_method=cluster_method)
        all_results[alpha] = res

        fss = _cfg.FSS_DICT.get(alpha, np.nan)
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
        save_dict[f"{key}_mean_P2"]          = pr["mean_P2"]
        save_dict[f"{key}_mean_abs_P1"]      = pr["mean_abs_P1"]
        save_dict[f"{key}_mean_abs_P1_sm"]   = pr["mean_abs_P1_sm"]
        save_dict[f"{key}_var_P1"]           = pr["var_P1"]
        save_dict[f"{key}_var_P1_sm"]        = pr["var_P1_sm"]
        save_dict[f"{key}_mean_abs_P2"]      = pr["mean_abs_P2"]
        save_dict[f"{key}_var_P2"]           = pr["var_P2"]
        save_dict[f"{key}_var_P2_sm"]        = pr["var_P2_sm"]
        save_dict[f"{key}_explained_var"]    = pr["explained_var"]
        save_dict[f"{key}_pc1_loading"]      = pr["pc1_loading"]
        save_dict[f"{key}_crossings_P1P2"]   = r["crossings_P1P2"]
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


# ── Alpha-sweep variants ──────────────────────────────────────────────────────
# These mirror run_pca / extract_beta_c / run_all but operate on a fixed-beta,
# varying-alpha file (load_alpha_sweep).  Statistics and clustering are computed
# as a function of alpha instead of beta, and the ordering convention is
# reversed: HIGH alpha = CDW (ordered) phase.

def run_pca_alpha_sweep(beta, file_template, n_components=10,
                        alpha_min=None, alpha_max=None):
    """
    Run PCA on alpha-sweep data (fixed beta, varying alpha).

    Returns
    -------
    dict with same keys as run_pca but keyed on alpha:
        'P', 'alphas', 'u_alpha',
        'mean_P1', 'mean_abs_P1', 'mean_abs_P1_sm',
        'var_P1', 'var_P1_sm', 'mean_abs_P2', 'var_P2', 'var_P2_sm',
        'explained_var'
    """
    X, alphas = load_alpha_sweep(beta, file_template=file_template,
                                 alpha_min=alpha_min, alpha_max=alpha_max)
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=n_components)
    P = pca.fit_transform(X_centered)

    u_alpha = np.unique(alphas)
    mean_P1     = np.array([np.mean(P[alphas == a, 0])         for a in u_alpha])
    mean_abs_P1 = np.array([np.mean(np.abs(P[alphas == a, 0])) for a in u_alpha])
    var_P1      = np.array([np.var(P[alphas == a, 0])           for a in u_alpha])
    mean_abs_P2 = np.array([np.mean(np.abs(P[alphas == a, 1])) for a in u_alpha])
    var_P2      = np.array([np.var(P[alphas == a, 1])           for a in u_alpha])
    mean_abs_P1_sm = gaussian_filter1d(mean_abs_P1, sigma=2)

    return {
        "P":               P,
        "alphas":          alphas,
        "u_alpha":         u_alpha,
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


def extract_alpha_c(beta, file_template, pca_result=None,
                    alpha_min=None, alpha_max=None, fss_alpha_c=None,
                    cluster_method="raw_2d"):
    """
    Extract critical alpha using PCA susceptibility and KMeans on alpha-sweep data.

    Parameters
    ----------
    fss_alpha_c : float or None
        Expected critical alpha (used as sigmoid fit initial guess).
        Defaults to the midpoint of u_alpha if not provided.

    Returns dict analogous to extract_beta_c, with keys:
        'argmax_P1', 'argmax_P2', 'sm_argmax_P1', 'sm_argmax_P2',
        'km_sig', 'km_sig_err', 'km_deriv', 'mean_P1_deriv_ac', 'best',
        'f_cdw_km', 'km_labels', 'km_sig_popt', 'pca_result',
        'var1_deriv_curve', 'var2_deriv_curve',
        'mean_P1_deriv_curve', 'km_deriv_curve'
    """
    if pca_result is None:
        pca_result = run_pca_alpha_sweep(beta, file_template,
                                         alpha_min=alpha_min, alpha_max=alpha_max)

    u_alpha        = pca_result["u_alpha"]
    var_P1         = pca_result["var_P1"]
    var_P1_sm      = pca_result["var_P1_sm"]
    var_P2         = pca_result["var_P2"]
    var_P2_sm      = pca_result["var_P2_sm"]
    mean_abs_P1_sm = pca_result["mean_abs_P1_sm"]
    P              = pca_result["P"]
    alphas_all     = pca_result["alphas"]

    if fss_alpha_c is None:
        fss_alpha_c = float(np.mean(u_alpha))

    # ── Argmax estimators ────────────────────────────────────────────────────
    ac_argmax_P1    = float(u_alpha[np.argmax(var_P1)])
    ac_argmax_P2    = float(u_alpha[np.argmax(var_P2)])
    ac_sm_argmax_P1 = float(u_alpha[np.argmax(var_P1_sm)])
    ac_sm_argmax_P2 = float(u_alpha[np.argmax(var_P2_sm)])

    # ── Derivative curves ────────────────────────────────────────────────────
    alpha_mid   = 0.5 * (u_alpha[:-1] + u_alpha[1:])
    d_var_sm1   = np.diff(var_P1_sm)   / np.diff(u_alpha)
    d_var_sm2   = np.diff(var_P2_sm)   / np.diff(u_alpha)
    d_mean_abs  = np.diff(mean_abs_P1_sm) / np.diff(u_alpha)
    ac_mean_P1_deriv = float(alpha_mid[np.argmax(d_mean_abs)])

    # ── KMeans: high alpha = CDW (ordered) phase ─────────────────────────────
    # PCA itself stays linear; clustering recipe selectable via cluster_method.
    km_labels = _cluster_pca_scores(P, method=cluster_method)

    mean_alpha_0 = np.mean(alphas_all[km_labels == 0])
    mean_alpha_1 = np.mean(alphas_all[km_labels == 1])
    cdw_label = 1 if mean_alpha_1 > mean_alpha_0 else 0

    f_cdw_km = np.array(
        [np.mean(km_labels[alphas_all == a] == cdw_label) for a in u_alpha]
    )

    try:
        popt_km, pcov_km = curve_fit(
            sigmoid_fit, u_alpha, f_cdw_km,
            p0=[fss_alpha_c, 0.3],
            bounds=([u_alpha[2], 0.01], [u_alpha[-3], 5.0]),
            maxfev=10000,
        )
        ac_km     = float(popt_km[0])
        ac_km_err = float(np.sqrt(pcov_km[0, 0]))
    except Exception:
        ac_km     = np.nan
        ac_km_err = np.nan
        popt_km   = None

    d_fkm       = np.diff(f_cdw_km) / np.diff(u_alpha)
    ac_km_deriv = float(alpha_mid[np.argmax(d_fkm)])

    return {
        "argmax_P1":           ac_argmax_P1,
        "argmax_P2":           ac_argmax_P2,
        "sm_argmax_P1":        ac_sm_argmax_P1,
        "sm_argmax_P2":        ac_sm_argmax_P2,
        "km_sig":              ac_km,
        "km_sig_err":          ac_km_err,
        "km_deriv":            ac_km_deriv,
        "mean_P1_deriv_ac":    ac_mean_P1_deriv,
        "best":                ac_sm_argmax_P1,
        "var1_deriv_curve":    (alpha_mid, d_var_sm1),
        "var2_deriv_curve":    (alpha_mid, d_var_sm2),
        "mean_P1_deriv_curve": (alpha_mid, d_mean_abs),
        "km_deriv_curve":      (alpha_mid, d_fkm),
        "km_labels":           km_labels,
        "f_cdw_km":            f_cdw_km,
        "km_sig_popt":         popt_km,
        "pca_result":          pca_result,
    }


def run_all_alpha_sweep(betas, file_template, alpha_min=None, alpha_max=None,
                        fss_alpha_c=None, tag=None, cluster_method="raw_2d"):
    """
    Run PCA alpha-sweep analysis for one or more fixed-beta files.
    Saves results to results/npz/pca_results_alpha_sweep[_tag].npz.

    `cluster_method` selects the post-PCA clustering recipe; see
    :func:`_cluster_pca_scores`.
    """
    results_dir = ensure_npz_dir()
    all_results = {}

    for beta in betas:
        print(f"[PCA alpha-sweep] beta={beta:.2f}  "
              f"cluster_method={cluster_method} ...")
        res = extract_alpha_c(beta, file_template,
                              alpha_min=alpha_min, alpha_max=alpha_max,
                              fss_alpha_c=fss_alpha_c,
                              cluster_method=cluster_method)
        all_results[beta] = res
        fss_str = f"{fss_alpha_c:.3f}" if fss_alpha_c is not None else "None"
        print(
            f"  sm_argmax_P1={res['sm_argmax_P1']:.3f}  "
            f"km_sig={res['km_sig']:.3f}±{res['km_sig_err']:.4f}  "
            f"km_deriv={res['km_deriv']:.3f}  FSS_alpha_c={fss_str}"
        )

    # Save
    save_dict = {}
    for beta in betas:
        r  = all_results[beta]
        pr = r["pca_result"]
        key = f"b{beta:.2f}"
        save_dict[f"{key}_u_alpha"]          = pr["u_alpha"]
        save_dict[f"{key}_mean_P1"]          = pr["mean_P1"]
        save_dict[f"{key}_mean_abs_P1"]      = pr["mean_abs_P1"]
        save_dict[f"{key}_mean_abs_P1_sm"]   = pr["mean_abs_P1_sm"]
        save_dict[f"{key}_var_P1"]           = pr["var_P1"]
        save_dict[f"{key}_var_P1_sm"]        = pr["var_P1_sm"]
        save_dict[f"{key}_mean_abs_P2"]      = pr["mean_abs_P2"]
        save_dict[f"{key}_var_P2"]           = pr["var_P2"]
        save_dict[f"{key}_var_P2_sm"]        = pr["var_P2_sm"]
        save_dict[f"{key}_explained_var"]    = pr["explained_var"]
        save_dict[f"{key}_alpha_c"] = np.array([
            r["argmax_P1"], r["argmax_P2"],
            r["km_sig"], r["km_sig_err"], r["km_deriv"], r["best"],
            r["mean_P1_deriv_ac"],
        ])
        save_dict[f"{key}_f_cdw_km"]         = r["f_cdw_km"]
        save_dict[f"{key}_km_deriv_alpha"]   = r["km_deriv_curve"][0]
        save_dict[f"{key}_km_deriv_vals"]    = r["km_deriv_curve"][1]
        if r["km_sig_popt"] is not None:
            save_dict[f"{key}_km_sig_popt"]  = np.array(r["km_sig_popt"])
        P      = pr["P"]
        alphas = pr["alphas"]
        idx = np.random.default_rng(42).choice(len(P), min(5000, len(P)), replace=False)
        save_dict[f"{key}_P1_sample"]        = P[idx, 0]
        save_dict[f"{key}_P2_sample"]        = P[idx, 1]
        save_dict[f"{key}_alphas_sample"]    = alphas[idx]
        save_dict[f"{key}_km_labels_sample"] = r["km_labels"][idx]

    suffix = f"_{tag}" if tag else "_alpha_sweep"
    fname  = f"pca_results{suffix}.npz"
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")
    return all_results


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
        run_all_alpha_sweep(
            betas=args.betas,
            file_template=file_template,
            fss_alpha_c=args.fss_alpha_c,
            tag=args.tag,
        )
    else:
        alphas        = args.alphas        or _cfg.ALPHAS
        file_template = args.file_template or _cfg.FILE_TEMPLATE
        run_all(alphas=alphas, beta_min=args.beta_min, beta_max=args.beta_max,
                stride=args.stride, file_template=file_template, tag=args.tag)


if __name__ == "__main__":
    main()
