"""
UMAP-based beta_c extraction.

How it works
------------
UMAP (Uniform Manifold Approximation and Projection) is a non-linear
dimensionality reduction method that preserves both local and global
structure better than t-SNE. The idea for phase detection:

1. **Embed** all configurations into 2D using UMAP. Configurations from
   the same phase cluster together because they share similar correlations.
   Near the transition, samples bridge the two clusters.

2. **Cluster** the 2D embedding into k=2 groups using KMeans (hard
   assignment — each point belongs to exactly one cluster).

3. **Identify phases**: the cluster whose members have higher average beta
   is labelled KVBS (ordered). The other is SM (disordered).

4. **Compute f_KVBS(beta)**: for each beta value, compute the fraction of
   samples assigned to the KVBS cluster. This transitions from ~0 (all SM
   at low beta) to ~1 (all KVBS at high beta) in a sigmoid shape.

5. **Fit sigmoid** f(beta) = 1 / (1 + exp(-(beta - beta_c) / w)) to
   extract beta_c as the midpoint of the transition.  The peak of the
   point-wise derivative df_KVBS/dbeta provides a second estimate.

Note: why we z-score the embedding before KMeans (`scale_kmeans=True`)
----------------------------------------------------------------------
KMeans minimises the within-cluster sum of squared Euclidean distances::

    inertia = Σ_k Σ_{i ∈ C_k} ||x_i - c_k||^2
            = Σ_k Σ_{i ∈ C_k} Σ_j (x_ij - c_kj)^2

Axis `j` contributes ~ N · V_j to the sum, where V_j is the variance along
that axis.  When the two UMAP coordinates have unequal spread, KMeans is
biased toward splitting along the wider axis even if the actual cluster
separation lies in another direction.

`StandardScaler().fit_transform(X)` computes per-column mean μ_j and std σ_j
and replaces each entry with x'_ij = (x_ij − μ_j) / σ_j, so every axis ends
up with mean 0 and std 1.  Each axis then contributes equally to KMeans's
distance metric, and the cluster boundary follows the genuine direction of
separation in the embedding.

Concretely on the attractive-Hubbard pilot (`L=8, U=−4, n=0.875`):

    observable                       (σ_x, σ_y)    ratio   effect of scaling
    integrated_pair_position         (~4.7, ~4.8)  0.98    near-isotropic; result barely changes
    integrated_pair_momentum         (~4.0, ~5.4)  0.74    pre-scale: KMeans biased toward emb_y,
                                                            f_KVBS non-monotonic (0 → 0.84 → 0.61);
                                                            post-scale: monotonic, lands at FSS β_c=8.5

What z-scoring does *not* do (important):
  * It does not remove outliers — a 20σ outlier stays 20σ in standardised
    space.  This is exactly why standardising raw PCA scores misbehaves on
    heavy-tailed observables: PC2's outlier tails become the dominant
    feature in the standardised plane and KMeans grabs *them* as a cluster.
  * It does not reshape distributions.  Heavy tails stay heavy, bimodal
    columns stay bimodal.
  * It is not a row-wise normalisation: it operates per column, not per
    sample, and does not put rows on a unit sphere.
  * Z-scoring before UMAP would change the embedding (UMAP's neighbour
    graph depends on the input metric).  We z-score *after* UMAP — the
    embedding itself is identical, only KMeans's per-axis weighting changes.

UMAP coords are roughly compact, bounded, and Gaussian-shaped (UMAP fits a
fuzzy neighbourhood graph and pulls outliers into the bulk rather than
projecting them to ±20σ), so per-column mean and std are meaningful and
z-scoring is appropriate.  PCA on heavy-tailed input data inherits the
heavy tails linearly, μ and σ are no longer summary statistics of "where
the typical data sits", and z-scoring of PCA scores is *not* recommended —
hence why this option lives in `umap_method.py` only.

Usage
-----
    python3.11 -m ssh_ml_phase.umap_method                                              # run all alphas
    python3.11 -m ssh_ml_phase.umap_method --alphas 0.240                               # single alpha
    python3.11 -m ssh_ml_phase.umap_method --beta-min 8.0                               # skip low-beta region
    python3.11 -m ssh_ml_phase.umap_method --file-template "GF_L12_a{alpha:.4f}.csv"    # new observable
    python3.11 -m ssh_ml_phase.umap_method --tag bmin8                                  # explicit output label
"""
import argparse
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import umap # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from . import config as _cfg
from .data import load_beta_sweep, load_alpha_sweep, ensure_npz_dir, make_tag, result_filename


def sigmoid_fit(beta, beta_c, w):
    """Standard logistic sigmoid: f = 1 / (1 + exp(-(beta - beta_c)/w))."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


def sigmoid_deriv(beta, beta_c, w):
    """Analytic derivative of the logistic sigmoid: df/dbeta = f(1-f)/w.
    Bell-shaped, peaks exactly at beta = beta_c with height 1/(4w)."""
    s = sigmoid_fit(beta, beta_c, w)
    return s * (1.0 - s) / w


def run_umap_analysis(alpha, n_neighbors=30, min_dist=0.3, pca_dims=50,
                      beta_min=None, beta_max=None, stride=None,
                      file_template=None, scale_kmeans=True):
    """
    Run UMAP + KMeans clustering for a single alpha value.

    Steps:
        1. Center the data and pre-reduce to `pca_dims` with PCA (for speed).
        2. UMAP to 2D.
        3. KMeans(k=2) on the 2D embedding.
        4. Sigmoid fit to f_KVBS(beta) and point-wise derivative.

    Parameters
    ----------
    scale_kmeans : bool, default True
        If True, z-score (StandardScaler) the 2D embedding before KMeans so
        the two axes contribute equally to Euclidean distance.  Recommended
        because skewed embeddings (e.g. integrated_pair_momentum at ~0.74:1)
        otherwise bias KMeans toward splitting along the wider axis.  Set
        False to cluster on the raw embedding (useful for diagnostic
        comparisons).

    Returns
    -------
    dict with keys:
        'embedding'       — 2D UMAP coordinates, shape (N, 2)
        'betas'           — beta label for every sample
        'u_beta'          — sorted unique beta values
        'km_labels'       — KMeans cluster assignments
        'f_kvbs_km'       — KVBS fraction at each beta
        'km_sig'          — beta_c from sigmoid fit
        'km_sig_err'      — 1-sigma error on km_sig from covariance matrix
        'km_deriv'        — beta_c from point-wise derivative peak of f_KVBS
        'best'            — km_sig (sigmoid peak has the right trend)
        'km_sig_popt'     — [beta_c, w] sigmoid fit parameters
        'km_deriv_curve'  — (beta_mid, d(f_KVBS)/dbeta) for visualization
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    X_centered = X - X.mean(axis=0)
    u_beta = np.unique(betas)
    fss_bc = _cfg.FSS_DICT.get(alpha, np.mean(u_beta))

    # PCA pre-reduction for speed
    pca = PCA(n_components=min(pca_dims, X.shape[1]))
    X_pca = pca.fit_transform(X_centered)

    # UMAP embedding
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=42,
    )
    embedding = reducer.fit_transform(X_pca)

    # ── KMeans clustering ──────────────────────────────────────────────────
    # Optionally z-score the 2D embedding so each axis contributes equally
    # to Euclidean distance — see the `scale_kmeans` doc above.
    km_input = StandardScaler().fit_transform(embedding) if scale_kmeans else embedding
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = km.fit_predict(km_input)

    # Identify which cluster is KVBS (ordered phase = higher avg beta)
    mean_beta_0 = np.mean(betas[km_labels == 0])
    mean_beta_1 = np.mean(betas[km_labels == 1])
    kvbs_label_km = 1 if mean_beta_1 > mean_beta_0 else 0

    f_kvbs_km = np.array(
        [np.mean(km_labels[betas == b] == kvbs_label_km) for b in u_beta]
    )

    # Sigmoid fit
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

    # Point-wise derivative of f_KVBS
    beta_mid    = 0.5 * (u_beta[:-1] + u_beta[1:])
    d_fkm       = np.diff(f_kvbs_km) / np.diff(u_beta)
    bc_km_deriv = float(beta_mid[np.argmax(d_fkm)])

    return {
        "embedding":      embedding,
        "betas":          betas,
        "u_beta":         u_beta,
        "km_labels":      km_labels,
        "f_kvbs_km":      f_kvbs_km,
        "km_sig":         bc_km,
        "km_sig_err":     bc_km_err,
        "km_deriv":       bc_km_deriv,
        "best":           bc_km,
        "km_sig_popt":    popt_km,
        "km_deriv_curve": (beta_mid, d_fkm),
    }


def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None, scale_kmeans=True):
    """
    Run UMAP analysis for all alphas. Saves to results/npz/umap_results.npz.

    beta_c array layout: [km_sig, km_sig_err, km_deriv, best]

    `scale_kmeans` (default True) toggles z-scoring of the 2D embedding
    before KMeans — see :func:`run_umap_analysis`.
    """
    if not HAS_UMAP:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        return {}

    if alphas is None:
        alphas = _cfg.ALPHAS

    results_dir = ensure_npz_dir()
    all_results = {}

    for alpha in alphas:
        print(f"[UMAP] alpha={alpha:.3f} ...")
        res = run_umap_analysis(alpha, beta_min=beta_min, beta_max=beta_max,
                                stride=stride, file_template=file_template,
                                scale_kmeans=scale_kmeans)
        all_results[alpha] = res

        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print(
            f"  km_sig={res['km_sig']:.1f}±{res['km_sig_err']:.2f}  "
            f"km_deriv={res['km_deriv']:.1f}  "
            f"best={res['best']:.1f}  FSS={fss:.1f}"
        )

    # Save
    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]        = r["u_beta"]
        save_dict[f"{key}_f_kvbs_km"]     = r["f_kvbs_km"]
        # beta_c: [km_sig, km_sig_err, km_deriv, best]
        save_dict[f"{key}_beta_c"] = np.array([
            r["km_sig"], r["km_sig_err"], r["km_deriv"], r["best"],
        ])
        save_dict[f"{key}_km_deriv_beta"] = r["km_deriv_curve"][0]
        save_dict[f"{key}_km_deriv_vals"] = r["km_deriv_curve"][1]
        if r["km_sig_popt"] is not None:
            save_dict[f"{key}_km_sig_popt"] = np.array(r["km_sig_popt"])
        # Subsampled embedding for scatter plots
        idx = np.random.default_rng(42).choice(
            len(r["embedding"]), min(5000, len(r["embedding"])), replace=False
        )
        save_dict[f"{key}_emb_x"]     = r["embedding"][idx, 0]
        save_dict[f"{key}_emb_y"]     = r["embedding"][idx, 1]
        save_dict[f"{key}_emb_betas"] = r["betas"][idx]
        save_dict[f"{key}_emb_km"]    = r["km_labels"][idx]

    fname = result_filename("umap", make_tag(beta_min, file_template, tag,
                                            beta_max=beta_max, stride=stride))
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


# ── Alpha-sweep variants ──────────────────────────────────────────────────────

def run_umap_alpha_sweep(beta, file_template, n_neighbors=30, min_dist=0.3,
                         pca_dims=50, alpha_min=None, alpha_max=None,
                         fss_alpha_c=None, scale_kmeans=True):
    """
    Run UMAP + KMeans on alpha-sweep data (fixed beta, varying alpha).

    Returns dict analogous to run_umap_analysis with keys:
        'embedding', 'alphas', 'u_alpha', 'km_labels', 'f_cdw_km',
        'km_sig', 'km_sig_err', 'km_deriv', 'best',
        'km_sig_popt', 'km_deriv_curve'
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    X, alphas = load_alpha_sweep(beta, file_template=file_template,
                                 alpha_min=alpha_min, alpha_max=alpha_max)
    X_centered = X - X.mean(axis=0)
    u_alpha = np.unique(alphas)

    if fss_alpha_c is None:
        fss_alpha_c = float(np.mean(u_alpha))

    # PCA pre-reduction for speed
    pca = PCA(n_components=min(pca_dims, X.shape[1]))
    X_pca = pca.fit_transform(X_centered)

    # UMAP embedding
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=42,
    )
    embedding = reducer.fit_transform(X_pca)

    # ── KMeans: high alpha = CDW (ordered) phase ──────────────────────────────
    # Optional z-scoring of the 2D embedding before clustering — see the
    # beta-sweep variant for the rationale.
    km_input = StandardScaler().fit_transform(embedding) if scale_kmeans else embedding
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = km.fit_predict(km_input)

    mean_alpha_0 = np.mean(alphas[km_labels == 0])
    mean_alpha_1 = np.mean(alphas[km_labels == 1])
    cdw_label = 1 if mean_alpha_1 > mean_alpha_0 else 0

    f_cdw_km = np.array(
        [np.mean(km_labels[alphas == a] == cdw_label) for a in u_alpha]
    )

    # Sigmoid fit
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

    alpha_mid   = 0.5 * (u_alpha[:-1] + u_alpha[1:])
    d_fkm       = np.diff(f_cdw_km) / np.diff(u_alpha)
    ac_km_deriv = float(alpha_mid[np.argmax(d_fkm)])

    return {
        "embedding":      embedding,
        "alphas":         alphas,
        "u_alpha":        u_alpha,
        "km_labels":      km_labels,
        "f_cdw_km":       f_cdw_km,
        "km_sig":         ac_km,
        "km_sig_err":     ac_km_err,
        "km_deriv":       ac_km_deriv,
        "best":           ac_km,
        "km_sig_popt":    popt_km,
        "km_deriv_curve": (alpha_mid, d_fkm),
    }


def run_all_alpha_sweep(betas, file_template, alpha_min=None, alpha_max=None,
                        fss_alpha_c=None, tag=None, scale_kmeans=True):
    """
    Run UMAP alpha-sweep analysis for one or more fixed-beta files.
    Saves to results/npz/umap_results_alpha_sweep[_tag].npz.

    alpha_c array layout per beta key: [km_sig, km_sig_err, km_deriv, best]

    `scale_kmeans` (default True) toggles z-scoring of the 2D embedding
    before KMeans — see :func:`run_umap_analysis`.
    """
    if not HAS_UMAP:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        return {}

    results_dir = ensure_npz_dir()
    all_results = {}

    for beta in betas:
        print(f"[UMAP alpha-sweep] beta={beta:.2f} ...")
        res = run_umap_alpha_sweep(beta, file_template,
                                   alpha_min=alpha_min, alpha_max=alpha_max,
                                   fss_alpha_c=fss_alpha_c,
                                   scale_kmeans=scale_kmeans)
        all_results[beta] = res
        fss_str = f"{fss_alpha_c:.3f}" if fss_alpha_c is not None else "None"
        print(
            f"  km_sig={res['km_sig']:.3f}±{res['km_sig_err']:.4f}  "
            f"km_deriv={res['km_deriv']:.3f}  FSS_alpha_c={fss_str}"
        )

    # Save
    save_dict = {}
    for beta in betas:
        r   = all_results[beta]
        key = f"b{beta:.2f}"
        save_dict[f"{key}_u_alpha"]        = r["u_alpha"]
        save_dict[f"{key}_f_cdw_km"]       = r["f_cdw_km"]
        save_dict[f"{key}_alpha_c"] = np.array([
            r["km_sig"], r["km_sig_err"], r["km_deriv"], r["best"],
        ])
        save_dict[f"{key}_km_deriv_alpha"] = r["km_deriv_curve"][0]
        save_dict[f"{key}_km_deriv_vals"]  = r["km_deriv_curve"][1]
        if r["km_sig_popt"] is not None:
            save_dict[f"{key}_km_sig_popt"] = np.array(r["km_sig_popt"])
        idx = np.random.default_rng(42).choice(
            len(r["embedding"]), min(5000, len(r["embedding"])), replace=False
        )
        save_dict[f"{key}_emb_x"]      = r["embedding"][idx, 0]
        save_dict[f"{key}_emb_y"]      = r["embedding"][idx, 1]
        save_dict[f"{key}_emb_alphas"] = r["alphas"][idx]
        save_dict[f"{key}_emb_km"]     = r["km_labels"][idx]

    suffix = f"_{tag}" if tag else "_alpha_sweep"
    fname  = f"umap_results{suffix}.npz"
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="UMAP-based beta_c extraction")
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
