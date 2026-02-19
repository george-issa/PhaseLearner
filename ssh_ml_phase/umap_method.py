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

Usage
-----
    python3.11 -m ssh_ml_phase.umap_method                   # run all alphas
    python3.11 -m ssh_ml_phase.umap_method --alphas 0.240    # single alpha
"""
import argparse
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import umap # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from .config import ALPHAS, FSS_DICT
from .data import load_data, ensure_results_dir


def sigmoid_fit(beta, beta_c, w):
    """Standard logistic sigmoid: f = 1 / (1 + exp(-(beta - beta_c)/w))."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


def sigmoid_deriv(beta, beta_c, w):
    """Analytic derivative of the logistic sigmoid: df/dbeta = f(1-f)/w.
    Bell-shaped, peaks exactly at beta = beta_c with height 1/(4w)."""
    s = sigmoid_fit(beta, beta_c, w)
    return s * (1.0 - s) / w


def run_umap_analysis(alpha, n_neighbors=30, min_dist=0.3, pca_dims=50):
    """
    Run UMAP + KMeans clustering for a single alpha value.

    Steps:
        1. Center the data and pre-reduce to `pca_dims` with PCA (for speed).
        2. UMAP to 2D.
        3. KMeans(k=2) on the 2D embedding.
        4. Sigmoid fit to f_KVBS(beta) and point-wise derivative.

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

    X, betas = load_data(alpha)
    X_centered = X - X.mean(axis=0)
    u_beta = np.unique(betas)
    fss_bc = FSS_DICT.get(alpha, np.mean(u_beta))

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
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = km.fit_predict(embedding)

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


def run_all(alphas=None):
    """
    Run UMAP analysis for all alphas. Saves to results/umap_results.npz.

    beta_c array layout: [km_sig, km_sig_err, km_deriv, best]
    """
    if not HAS_UMAP:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        return {}

    if alphas is None:
        alphas = ALPHAS

    results_dir = ensure_results_dir()
    all_results = {}

    for alpha in alphas:
        print(f"[UMAP] alpha={alpha:.3f} ...")
        res = run_umap_analysis(alpha)
        all_results[alpha] = res

        fss = FSS_DICT.get(alpha, np.nan)
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

    np.savez(f"{results_dir}/umap_results.npz", **save_dict)
    print(f"\nResults saved to {results_dir}/umap_results.npz")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="UMAP-based beta_c extraction")
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=None,
        help="Alpha values to process (default: all four)"
    )
    args = parser.parse_args()
    run_all(alphas=args.alphas)


if __name__ == "__main__":
    main()
