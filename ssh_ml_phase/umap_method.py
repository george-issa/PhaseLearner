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

2. **Cluster** the 2D embedding into k=2 groups using:
   - KMeans (hard assignment — each point belongs to exactly one cluster)
   - Gaussian Mixture Model (soft assignment — gives a probability for each cluster)

3. **Identify phases**: the cluster whose members have higher average beta
   is labelled KVBS (ordered). The other is SM (disordered).

4. **Compute f_KVBS(beta)**: for each beta value, compute the fraction of
   samples assigned to (or with high probability of) the KVBS cluster.
   This fraction transitions from ~0 (all SM at low beta) to ~1 (all KVBS
   at high beta) in a sigmoid shape.

5. **Fit sigmoid** f(beta) = 1 / (1 + exp(-(beta - beta_c) / w)) to
   extract beta_c as the midpoint of the transition.

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
from sklearn.mixture import GaussianMixture

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


def run_umap_analysis(alpha, n_neighbors=30, min_dist=0.3, pca_dims=50):
    """
    Run UMAP + clustering for a single alpha value.

    Steps:
        1. Center the data and pre-reduce to `pca_dims` with PCA (for speed).
        2. UMAP to 2D.
        3. KMeans(k=2) and GMM(k=2) on the 2D embedding.
        4. Sigmoid fits to f_KVBS(beta).

    Returns
    -------
    dict with embedding, cluster labels, beta_c estimates, etc.
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

    # f_KVBS(beta) from KMeans
    f_kvbs_km = np.array(
        [np.mean(km_labels[betas == b] == kvbs_label_km) for b in u_beta]
    )

    try:
        popt_km, _ = curve_fit(sigmoid_fit, u_beta, f_kvbs_km,
                               p0=[fss_bc, 2.0],
                               bounds=([u_beta[2], 0.1], [u_beta[-3], 20.0]),
                               maxfev=10000)
        bc_km = popt_km[0]
    except Exception:
        bc_km = np.nan
        popt_km = None

    # Derivative of f_KVBS_km → peak = steepest change = transition
    d_fkm = np.diff(f_kvbs_km) / np.diff(u_beta)
    beta_mid = 0.5 * (u_beta[:-1] + u_beta[1:])
    bc_km_deriv = beta_mid[np.argmax(d_fkm)]

    # ── GMM clustering ─────────────────────────────────────────────────────
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(embedding)
    gmm_probs = gmm.predict_proba(embedding)

    mean_beta_g0 = np.mean(betas[gmm_labels == 0])
    mean_beta_g1 = np.mean(betas[gmm_labels == 1])
    kvbs_label_gmm = 1 if mean_beta_g1 > mean_beta_g0 else 0

    # f_KVBS(beta) from GMM (soft probabilities)
    f_kvbs_gmm = np.array(
        [np.mean(gmm_probs[betas == b, kvbs_label_gmm]) for b in u_beta]
    )

    try:
        popt_gmm, _ = curve_fit(sigmoid_fit, u_beta, f_kvbs_gmm,
                                p0=[fss_bc, 2.0],
                                bounds=([u_beta[2], 0.1], [u_beta[-3], 20.0]),
                                maxfev=10000)
        bc_gmm = popt_gmm[0]
    except Exception:
        bc_gmm = np.nan
        popt_gmm = None

    # Derivative of f_KVBS_gmm
    d_fgmm = np.diff(f_kvbs_gmm) / np.diff(u_beta)
    bc_gmm_deriv = beta_mid[np.argmax(d_fgmm)]

    # Best estimate: derivative of KMeans f_KVBS (most stable)
    bc_best = bc_km_deriv

    return {
        "embedding": embedding,
        "betas": betas,
        "u_beta": u_beta,
        "km_labels": km_labels,
        "gmm_labels": gmm_labels,
        "gmm_probs": gmm_probs,
        "f_kvbs_km": f_kvbs_km,
        "f_kvbs_gmm": f_kvbs_gmm,
        "beta_c_km": bc_km,
        "beta_c_gmm": bc_gmm,
        "beta_c_km_deriv": bc_km_deriv,
        "beta_c_gmm_deriv": bc_gmm_deriv,
        "beta_c_best": bc_best,
        "sigmoid_params_km": popt_km,
        "sigmoid_params_gmm": popt_gmm,
        "deriv_km": (beta_mid, d_fkm),
        "deriv_gmm": (beta_mid, d_fgmm),
    }


def run_all(alphas=None):
    """
    Run UMAP analysis for all alphas. Saves to results/umap_results.npz.
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
            f"  KMeans: sig={res['beta_c_km']:.1f} deriv={res['beta_c_km_deriv']:.1f}  "
            f"GMM: sig={res['beta_c_gmm']:.1f} deriv={res['beta_c_gmm_deriv']:.1f}  "
            f"best={res['beta_c_best']:.1f}  FSS={fss:.1f}"
        )

    # Save
    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"] = r["u_beta"]
        save_dict[f"{key}_f_kvbs_km"] = r["f_kvbs_km"]
        save_dict[f"{key}_f_kvbs_gmm"] = r["f_kvbs_gmm"]
        save_dict[f"{key}_beta_c"] = np.array([
            r["beta_c_km"], r["beta_c_gmm"],
            r["beta_c_km_deriv"], r["beta_c_gmm_deriv"], r["beta_c_best"]
        ])
        save_dict[f"{key}_deriv_km_beta"] = r["deriv_km"][0]
        save_dict[f"{key}_deriv_km_vals"] = r["deriv_km"][1]
        save_dict[f"{key}_deriv_gmm_vals"] = r["deriv_gmm"][1]
        if r["sigmoid_params_km"] is not None:
            save_dict[f"{key}_sigmoid_km"] = np.array(r["sigmoid_params_km"])
        if r["sigmoid_params_gmm"] is not None:
            save_dict[f"{key}_sigmoid_gmm"] = np.array(r["sigmoid_params_gmm"])
        # Save subsampled embedding for scatter plots
        idx = np.random.default_rng(42).choice(
            len(r["embedding"]), min(5000, len(r["embedding"])), replace=False
        )
        save_dict[f"{key}_emb_x"] = r["embedding"][idx, 0]
        save_dict[f"{key}_emb_y"] = r["embedding"][idx, 1]
        save_dict[f"{key}_emb_betas"] = r["betas"][idx]
        save_dict[f"{key}_emb_km"] = r["km_labels"][idx]

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
