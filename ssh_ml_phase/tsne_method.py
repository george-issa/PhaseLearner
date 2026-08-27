"""
t-SNE based β_c extraction.

Same clustering idea as the UMAP method, only the 2D embedder changes:

  1. Embed every DQMC configuration into 2D with t-SNE.
  2. Split the cloud into two clusters with KMeans (k=2).
  3. Call the cluster with the higher average β the "ordered" phase.
  4. For each β, measure the fraction of configs in the ordered cluster —
     this rises from ~0 (disordered) to ~1 (ordered) as β grows.
  5. β_c is the midpoint of a sigmoid fit to that fraction curve.

Results are saved as ``results/npz/tsne_results_<tag>.npz`` using the same
key layout as the UMAP method, so the existing plotter renders them too.
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from . import config as _cfg
from .data import (load_beta_sweep, ensure_npz_dir, make_tag, result_filename,
                   lattice_snapshots, store_snapshots)


def sigmoid(beta, beta_c, w):
    """Logistic step from 0 to 1, centred at beta_c with width w."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


def embed_and_cluster(alpha, *, beta_min=None, beta_max=None, stride=None,
                      file_template=None, perplexity=30, pca_dims=50):
    """Run the five-step t-SNE → β_c analysis for a single alpha.

    Returns a dict with the embedding, the fraction-ordered curve, and the
    two β_c estimates (sigmoid midpoint and steepest-slope point).
    """
    # 1. Load every configuration for this alpha; `betas` labels each row.
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    u_beta = np.unique(betas)

    # 2. t-SNE is slow in high dimensions, so pre-reduce with PCA, then
    #    embed down to 2D.  init="pca" makes the layout reproducible.
    X_pca = PCA(n_components=min(pca_dims, X.shape[1])).fit_transform(X - X.mean(0))
    embedding = TSNE(n_components=2, perplexity=perplexity,
                     init="pca", random_state=42).fit_transform(X_pca)

    # 3. Cluster the 2D cloud into two groups.
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(embedding)

    # 4. The cluster sitting at higher average β is the ordered phase.
    #    f_ordered[i] = fraction of configs at β=u_beta[i] in that cluster.
    ordered = 1 if betas[labels == 1].mean() > betas[labels == 0].mean() else 0
    f_ordered = np.array([(labels[betas == b] == ordered).mean() for b in u_beta])

    # 5. β_c from a sigmoid fit (midpoint) and from the steepest finite
    #    difference of the fraction curve (a fit-free cross-check).
    fss_bc = _cfg.FSS_DICT.get(alpha, float(u_beta.mean()))
    try:
        popt, pcov = curve_fit(sigmoid, u_beta, f_ordered, p0=[fss_bc, 2.0],
                               bounds=([u_beta[2], 0.1], [u_beta[-3], 20.0]),
                               maxfev=10000)
        beta_c, beta_c_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        beta_c, beta_c_err, popt = np.nan, np.nan, None

    beta_mid = 0.5 * (u_beta[:-1] + u_beta[1:])
    slope = np.diff(f_ordered) / np.diff(u_beta)
    beta_c_deriv = float(beta_mid[np.argmax(slope)])

    inlat_betas, inlat_grids = lattice_snapshots(X, betas)

    return {
        "u_beta": u_beta, "betas": betas, "embedding": embedding,
        "labels": labels, "f_ordered": f_ordered,
        "beta_c": beta_c, "beta_c_err": beta_c_err, "beta_c_deriv": beta_c_deriv,
        "popt": popt, "slope_curve": (beta_mid, slope),
        "inlat_betas": inlat_betas, "inlat_grids": inlat_grids,
    }


def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None, perplexity=30):
    """Run the t-SNE analysis for every alpha and save one .npz of results.

    ``perplexity`` is t-SNE's neighborhood size (typically 5–50); larger values
    weight broader structure.
    """
    alphas = list(alphas) if alphas else list(_cfg.ALPHAS)
    results_dir = ensure_npz_dir()
    results, save = {}, {}

    for alpha in alphas:
        print(f"[t-SNE] alpha={alpha:.3f} (perplexity={perplexity}) ...")
        r = embed_and_cluster(alpha, beta_min=beta_min, beta_max=beta_max,
                              stride=stride, file_template=file_template,
                              perplexity=perplexity)
        results[alpha] = r
        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print(f"  beta_c={r['beta_c']:.1f}±{r['beta_c_err']:.2f}  "
              f"deriv={r['beta_c_deriv']:.1f}  FSS={fss:.1f}")

        # Store under the same keys the UMAP plotter reads (km_sig == sigmoid
        # midpoint), so t-SNE plots for free via plot_umap_beta_sweep.
        key = f"a{alpha:.4f}"
        save[f"{key}_u_beta"]        = r["u_beta"]
        save[f"{key}_f_kvbs_km"]     = r["f_ordered"]
        save[f"{key}_beta_c"]        = np.array([r["beta_c"], r["beta_c_err"],
                                                 r["beta_c_deriv"], r["beta_c"]])
        save[f"{key}_km_deriv_beta"] = r["slope_curve"][0]
        save[f"{key}_km_deriv_vals"] = r["slope_curve"][1]
        if r["popt"] is not None:
            save[f"{key}_km_sig_popt"] = np.array(r["popt"])
        # Subsample the embedding so the scatter-plot keys stay small.
        idx = np.random.default_rng(42).choice(
            len(r["embedding"]), min(5000, len(r["embedding"])), replace=False)
        save[f"{key}_emb_x"]     = r["embedding"][idx, 0]
        save[f"{key}_emb_y"]     = r["embedding"][idx, 1]
        save[f"{key}_emb_betas"] = r["betas"][idx]
        save[f"{key}_emb_km"]    = r["labels"][idx]
        store_snapshots(save, key, r)

    fname = result_filename("tsne", make_tag(beta_min, file_template, tag,
                                             beta_max=beta_max, stride=stride))
    np.savez(f"{results_dir}/{fname}", **save)
    print(f"\nResults saved to {results_dir}/{fname}")
    return results
