"""
t-SNE embedding of DQMC configurations, with beta_c read from the CLUSTER
STRUCTURE of the map -- not from a two-way split of it.

Why the old version was replaced (2026-09-06, obsidian/07_Failures_and_Negative_Results.md):
KMeans(k=2) on a t-SNE map splits the sample set at its median beta whenever the
map is a beta-ordered chain, which it is for every smooth observable (and, because
t-SNE equalises density, even for the 2D Ising model).  The "beta_c" that came out
was the midpoint of the beta grid.  The grids were centred on the known beta_c, so
the artifact looked like agreement.

What this module does instead:

  1. Embed every configuration with PCA(50) -> t-SNE.  Perplexity and a
     per-beta subsample are settings, and both are recorded.
  2. Diagnose the map.  ``worm_rho`` is |Spearman| between the map's principal
     axis and beta: ~1 means a beta-ordered chain with no phase structure.
  3. Find density clusters with HDBSCAN.  If fewer than two clusters exist the
     result is "no cluster structure" (NaN), never a number.
  4. Read beta_c from the extent of the clusters (the readings the user asked
     for): the lowest beta present in the ordered (high-beta) cluster, the highest
     beta present in the disordered (low-beta) cluster, and the onset beta at
     which the ordered cluster first holds half of a beta's clustered samples.
     Percentiles (5 % / 95 %) stand in for min / max so single outliers do not
     move the reading.  The readings are repeated over a sweep of HDBSCAN's
     minimum cluster size; their spread is the reported uncertainty.
     These turned out NOT to survive the truncation test on any SSH observable
     (the clusters themselves change with the window); they are kept as
     diagnostics.
  4b. The NECK reading -- the one map-shape reading that does survive it.
     Maps of the equal-time BondBond012345 correlator are horseshoes: a thin
     low-beta arc and a diffuse high-beta cloud joined by a neck.  n(beta) is
     the number of map points of ANY beta within r = 0.5 x (mean within-beta
     spread) of the beta centroid, divided by its median along the path; the
     neck is the interior minimum.  At the neck beta a configuration is equally
     likely to sit in either lobe.  Needs both lobes well inside the window.
  5. The KMeans bisection is kept only as a flagged legacy diagnostic so the
     existing comparison scripts still find their keys.  It is seeded with the
     grid midpoint -- FSS never enters any estimator here.
  6. The embedding figure (map by beta, map by cluster, membership curves) is
     always written, because whether a map supports a reading is a judgement
     that has to be made by looking at it.

Validation rule: a reading is only reportable if it stays put when the beta
window is truncated asymmetrically -- see ``truncation_test``.

Results: ``results/npz/tsne_results_<tag>.npz`` (legacy keys unchanged, new
``*_cluster_*`` / ``*_worm_rho`` keys added); figure under
``results/figures/<model>/tsne/<tag>/embedding_a<alpha>.png``.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from . import config as _cfg
from .data import (load_beta_sweep, ensure_npz_dir, make_tag, result_filename,
                   lattice_snapshots, store_snapshots)

MIN_CLUSTER_FRACS = (0.02, 0.04, 0.06, 0.10)   # HDBSCAN min_cluster_size as a fraction of N
MIN_SAMPLES = 20


def sigmoid(beta, beta_c, w):
    """Logistic step from 0 to 1, centred at beta_c with width w."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


# ---------------------------------------------------------------------------
# 1. embedding
# ---------------------------------------------------------------------------
def embed(X, perplexity=30, pca_dims=50, seed=42):
    """PCA pre-reduction, then 2D t-SNE.  init='pca' makes the layout reproducible."""
    X_pca = PCA(n_components=min(pca_dims, X.shape[1])).fit_transform(X - X.mean(0))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed, n_jobs=-1).fit_transform(X_pca)


# ---------------------------------------------------------------------------
# 2. diagnostics
# ---------------------------------------------------------------------------
def diagnose(E, betas):
    """How much of the map is just beta?

    worm_rho     |Spearman| between the map's principal axis and beta.
    step_ratio   median distance between consecutive per-beta centroids divided
                 by the mean within-beta spread; >1 means the beta groups are
                 laid out as separate beads on a chain.
    """
    axis = PCA(n_components=1).fit_transform(E)[:, 0]
    worm_rho = abs(spearmanr(axis, betas).correlation)
    u = np.unique(betas)
    centroids = np.array([E[betas == b].mean(0) for b in u])
    spread = np.mean([np.sqrt(((E[betas == b] - c) ** 2).sum(1).mean())
                      for b, c in zip(u, centroids)])
    steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    return {"worm_rho": float(worm_rho), "centroids": centroids,
            "step_ratio": float(np.median(steps) / spread)}


def neck_from_curve(u, n, edge=2, min_prominence=0.1):
    """Locate the neck on a density-along-path curve n(beta).

    The neck is the deepest interior LOCAL minimum whose prominence (how far the
    curve rises again on both sides) is at least ``min_prominence`` in units of the
    median-normalised density.  A curve that merely slopes down into a sparse tail
    has no such minimum and returns (nan, nan): "no neck", not a number.  The
    first/last ``edge`` grid points are excluded.
    """
    from scipy.signal import find_peaks
    inner = slice(edge, len(u) - edge)
    idx, _ = find_peaks(-n[inner], prominence=min_prominence)
    if len(idx) == 0:
        return np.nan, np.nan
    i = int(idx[np.argmin(n[inner][idx])]) + edge
    return float(u[i]), float(n[i])


def neck_reading(E, betas, r_frac=0.5, edge=2):
    """Density of the map along the per-beta centroid path, and its neck.

    Returns (u_beta, n_curve, neck_beta, neck_depth).  ``n_curve`` is the count of
    points of any beta within ``r_frac`` x (mean within-beta spread) of each beta
    centroid, divided by its median; ``neck_depth`` is the curve's value at the
    neck (1 = no dip).  See ``neck_from_curve`` for what counts as a neck.
    """
    from scipy.spatial import cKDTree
    u = np.unique(betas)
    centroids = np.array([E[betas == b].mean(0) for b in u])
    spread = np.mean([np.sqrt(((E[betas == b] - c) ** 2).sum(1).mean())
                      for b, c in zip(u, centroids)])
    tree = cKDTree(E)
    n = np.array([len(tree.query_ball_point(c, r_frac * spread)) for c in centroids], float)
    n /= np.median(n)
    # Guard: a neck is only meaningful when consecutive beta clouds overlap.  On a
    # bead chain (tau-integrated snakes) the centroids sit in empty space between
    # beads, the path density is zero there, and the "neck" is meaningless.
    steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    if np.median(steps) > spread:
        return u, n, np.nan, np.nan
    neck_beta, neck_depth = neck_from_curve(u, n, edge=edge)
    return u, n, neck_beta, neck_depth


# ---------------------------------------------------------------------------
# 3 + 4. density clusters and the readings taken from their extent
# ---------------------------------------------------------------------------
def cluster_readings(E, betas, min_cluster_frac=0.06, min_samples=MIN_SAMPLES):
    """HDBSCAN on the z-scored map, then beta_c readings from cluster extent.

    Returns a dict with
        labels         HDBSCAN labels (-1 = noise)
        n_clusters
        beta_lo_high   lowest beta (5th percentile) in the highest-beta cluster
        beta_hi_low    highest beta (95th percentile) in the lowest-beta cluster
        onset          first beta at which the high cluster holds >= 50 % of that
                       beta's clustered samples
        f_high         per-beta fraction of clustered samples in the high cluster
    All readings are NaN when fewer than two clusters are found.
    """
    mcs = max(50, int(min_cluster_frac * len(betas)))
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples).fit_predict(
        StandardScaler().fit_transform(E))
    ids = [i for i in np.unique(labels) if i >= 0]
    u = np.unique(betas)
    out = {"labels": labels, "n_clusters": len(ids), "beta_lo_high": np.nan,
           "beta_hi_low": np.nan, "onset": np.nan, "f_high": np.full(len(u), np.nan),
           "min_cluster_size": mcs}
    if len(ids) < 2:
        return out

    median_beta = {i: np.median(betas[labels == i]) for i in ids}
    low, high = min(ids, key=median_beta.get), max(ids, key=median_beta.get)
    clustered = labels >= 0
    f_high = np.array([np.mean(labels[(betas == b) & clustered] == high)
                       if np.any((betas == b) & clustered) else np.nan for b in u])
    out.update({
        "beta_lo_high": float(np.percentile(betas[labels == high], 5)),
        "beta_hi_low": float(np.percentile(betas[labels == low], 95)),
        "onset": float(u[np.argmax(f_high >= 0.5)]) if np.any(f_high >= 0.5) else np.nan,
        "f_high": f_high, "low": low, "high": high,
    })
    return out


def readings_with_spread(E, betas, fracs=MIN_CLUSTER_FRACS):
    """Readings at each min-cluster-size setting; the spread is the uncertainty."""
    per_frac = {f: cluster_readings(E, betas, min_cluster_frac=f) for f in fracs}
    spread = {}
    for name in ("beta_lo_high", "beta_hi_low", "onset"):
        vals = np.array([r[name] for r in per_frac.values()], dtype=float)
        spread[name] = (np.nanmin(vals), np.nanmax(vals)) if np.isfinite(vals).any() else (np.nan, np.nan)
    return per_frac, spread


# ---------------------------------------------------------------------------
# 5. legacy bisection (diagnostic only)
# ---------------------------------------------------------------------------
def kmeans_bisection(E, betas):
    """KMeans(k=2) ordered fraction + sigmoid midpoint.  LEGACY / DIAGNOSTIC:
    on a beta-ordered map this returns the median beta of the window, not a
    transition.  Seeded with the grid midpoint, never with FSS."""
    u = np.unique(betas)
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(E)
    ordered = 1 if betas[labels == 1].mean() > betas[labels == 0].mean() else 0
    f = np.array([(labels[betas == b] == ordered).mean() for b in u])
    try:
        popt, pcov = curve_fit(sigmoid, u, f, p0=[0.5 * (u[0] + u[-1]), 2.0],
                               bounds=([u[2], 0.1], [u[-3], 20.0]), maxfev=10000)
        beta_c, err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        beta_c, err, popt = np.nan, np.nan, None
    beta_mid = 0.5 * (u[:-1] + u[1:])
    slope = np.diff(f) / np.diff(u)
    return {"labels": labels, "f": f, "beta_c": beta_c, "beta_c_err": err,
            "beta_c_deriv": float(beta_mid[np.argmax(slope)]), "popt": popt,
            "slope_curve": (beta_mid, slope), "share": float((labels == ordered).mean())}


# ---------------------------------------------------------------------------
# one alpha
# ---------------------------------------------------------------------------
def embed_and_cluster(alpha, *, beta_min=None, beta_max=None, stride=None,
                      file_template=None, perplexity=30, pca_dims=50, seed=42):
    """Embed, diagnose, cluster and read beta_c for a single alpha."""
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    E = embed(X, perplexity=perplexity, pca_dims=pca_dims, seed=seed)
    diag = diagnose(E, betas)
    per_frac, spread = readings_with_spread(E, betas)
    main = per_frac[0.06]
    _, neck_curve, neck_beta, neck_depth = neck_reading(E, betas)
    legacy = kmeans_bisection(E, betas)
    inlat_betas, inlat_grids = lattice_snapshots(X, betas)
    return {
        "u_beta": np.unique(betas), "betas": betas, "embedding": E,
        "perplexity": perplexity, "samples_per_beta": int(np.bincount(
            np.unique(betas, return_inverse=True)[1]).mean()),
        **diag, "readings": main, "readings_per_frac": per_frac, "spread": spread,
        "neck_curve": neck_curve, "neck_beta": neck_beta, "neck_depth": neck_depth,
        "legacy": legacy,
        "inlat_betas": inlat_betas, "inlat_grids": inlat_grids,
    }


def describe(r, fss=np.nan):
    """One-line summary of an alpha's result (FSS shown for reference only)."""
    rd, sp = r["readings"], r["spread"]
    def fmt(name):
        lo, hi = sp[name]
        return f"{rd[name]:5.1f} [{lo:.1f}-{hi:.1f}]"
    if rd["n_clusters"] < 2:
        cl = "NO CLUSTER STRUCTURE"
    else:
        cl = (f"beta_lo(high cl)={fmt('beta_lo_high')}  beta_hi(low cl)={fmt('beta_hi_low')}"
              f"  onset={fmt('onset')}")
    neck = (f"NECK beta={r['neck_beta']:.1f} (depth {r['neck_depth']:.2f})"
            if np.isfinite(r["neck_beta"]) else "NECK: none (no prominent dip)")
    return (f"{neck}  | "
            f"worm_rho={r['worm_rho']:.2f} step_ratio={r['step_ratio']:.2f} "
            f"clusters={rd['n_clusters']}  {cl}  | legacy km_sig={r['legacy']['beta_c']:.1f} "
            f"(share {r['legacy']['share']:.2f})  | FSS={fss:.1f}")


# ---------------------------------------------------------------------------
# 6. figure
# ---------------------------------------------------------------------------
def plot_embedding(r, alpha, path, fss=np.nan, title=""):
    """Three panels: map coloured by beta, map coloured by HDBSCAN cluster,
    and the per-beta membership curves with the readings marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    E, b, u = r["embedding"], r["betas"], r["u_beta"]
    idx = np.random.default_rng(0).permutation(len(b))[:8000]
    rd, sp = r["readings"], r["spread"]
    # plain-text rendering: the paper style module turns usetex on globally
    with matplotlib.rc_context({"text.usetex": False, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
        _draw_embedding_panels(fig, ax, r, alpha, fss, title, E, b, u, idx, rd, sp)
        ax[3].plot(u, r["neck_curve"], "o-", ms=3, color="C0")
        if np.isfinite(r["neck_beta"]):
            ax[3].axvline(r["neck_beta"], color="C0", lw=1.2,
                          label=f"neck = {r['neck_beta']:.1f} (depth {r['neck_depth']:.2f})")
        else:
            ax[3].text(0.5, 0.5, "no prominent dip: no neck", transform=ax[3].transAxes,
                       ha="center", va="center", fontsize=10, color="0.4")
        if np.isfinite(fss):
            ax[3].axvline(fss, color="k", lw=1, ls=":", label=f"FSS beta_c = {fss:.1f} (reference)")
        ax[3].set_xlabel("beta"); ax[3].set_ylabel("map density along centroid path / median")
        ax[3].legend(fontsize=7, loc="best"); ax[3].grid(alpha=0.3)
        ax[3].set_title("neck function n(beta): points of any beta within 0.5 spread of the centroid", fontsize=9)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _draw_embedding_panels(fig, ax, r, alpha, fss, title, E, b, u, idx, rd, sp):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    centre = fss if np.isfinite(fss) and u[0] < fss < u[-1] else 0.5 * (u[0] + u[-1])
    norm = TwoSlopeNorm(vmin=u[0], vcenter=centre, vmax=u[-1])
    sc = ax[0].scatter(E[idx, 0], E[idx, 1], c=b[idx], cmap="coolwarm_r", norm=norm, s=2, lw=0)
    c = r["centroids"]
    ax[0].plot(c[:, 0], c[:, 1], "k-", lw=0.6, alpha=0.7)
    ax[0].plot(c[0, 0], c[0, 1], "ko", ms=4, label=f"beta={u[0]:g}")
    ax[0].plot(c[-1, 0], c[-1, 1], "ks", ms=4, label=f"beta={u[-1]:g}")
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02); cb.set_label("beta")
    if np.isfinite(fss):
        cb.ax.axhline(fss, color="k", lw=1)
    ax[0].legend(fontsize=7, loc="best")
    ax[0].set_title(f"{title} alpha={alpha:.4f}: by beta   worm_rho={r['worm_rho']:.2f}, "
                    f"step_ratio={r['step_ratio']:.2f}", fontsize=9)

    lab = rd["labels"]
    ax[1].scatter(E[idx][lab[idx] < 0, 0], E[idx][lab[idx] < 0, 1], c="0.85", s=2, lw=0)
    ids = [i for i in np.unique(lab) if i >= 0]
    cmap = plt.get_cmap("tab10")
    for k, i in enumerate(sorted(ids, key=lambda i: np.median(b[lab == i]))):
        m = (lab == i)[idx]
        q = np.percentile(b[lab == i], [5, 50, 95])
        ax[1].scatter(E[idx][m, 0], E[idx][m, 1], color=cmap(k % 10), s=2, lw=0,
                      label=f"beta {q[0]:.1f}-{q[2]:.1f} (median {q[1]:.1f}, n={np.sum(lab == i)})")
    if ids:
        ax[1].legend(fontsize=7, loc="best", markerscale=4)
    ax[1].set_title(f"HDBSCAN (min size {rd['min_cluster_size']}): {rd['n_clusters']} clusters, "
                    f"noise {(lab < 0).mean():.0%}", fontsize=9)
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])

    ax[2].plot(u, rd["f_high"], "o-", ms=3, label="HDBSCAN: fraction in high-beta cluster")
    ax[2].plot(u, r["legacy"]["f"], "s--", ms=3, color="0.5", label="legacy KMeans bisection")
    for name, col in (("beta_lo_high", "C1"), ("beta_hi_low", "C2"), ("onset", "C3")):
        if np.isfinite(rd[name]):
            ax[2].axvline(rd[name], color=col, lw=1, label=f"{name} = {rd[name]:.1f} "
                          f"[{sp[name][0]:.1f}-{sp[name][1]:.1f}]")
    if np.isfinite(fss):
        ax[2].axvline(fss, color="k", lw=1, ls=":", label=f"FSS beta_c = {fss:.1f} (reference)")
    ax[2].set_xlabel("beta"); ax[2].set_ylabel("fraction"); ax[2].set_ylim(-0.02, 1.02)
    ax[2].legend(fontsize=7, loc="best"); ax[2].set_title("membership curves and readings", fontsize=9)


def _figure_path(tag, alpha, model=None):
    from .plot import _figures_dir
    return f"{_figures_dir(tag, 'tsne', model)}/embedding_a{alpha:.4f}.png"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_all(alphas=None, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None, perplexity=30, plot=True, model=None,
            seed=42):
    """Run the t-SNE analysis for every alpha, save one .npz, write the figures."""
    alphas = list(alphas) if alphas else list(_cfg.ALPHAS)
    results_dir = ensure_npz_dir()
    tag = make_tag(beta_min, file_template, tag, beta_max=beta_max, stride=stride)
    results, save = {}, {}

    for alpha in alphas:
        print(f"[t-SNE] alpha={alpha:.3f} (perplexity={perplexity}, seed={seed}) ...", flush=True)
        r = embed_and_cluster(alpha, beta_min=beta_min, beta_max=beta_max,
                              stride=stride, file_template=file_template,
                              perplexity=perplexity, seed=seed)
        results[alpha] = r
        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print("  " + describe(r, fss), flush=True)
        if plot:
            path = _figure_path(tag, alpha, model)
            plot_embedding(r, alpha, path, fss=fss, title=tag or "")
            print(f"  figure: {path}", flush=True)

        key = f"a{alpha:.4f}"
        leg, rd, sp = r["legacy"], r["readings"], r["spread"]
        # legacy keys (same layout as before, so plot_umap_beta_sweep and
        # compare_methods_ssh keep working) -- these are the bisection numbers
        save[f"{key}_u_beta"]        = r["u_beta"]
        save[f"{key}_f_kvbs_km"]     = leg["f"]
        save[f"{key}_beta_c"]        = np.array([leg["beta_c"], leg["beta_c_err"],
                                                 leg["beta_c_deriv"], leg["beta_c"]])
        save[f"{key}_km_deriv_beta"] = leg["slope_curve"][0]
        save[f"{key}_km_deriv_vals"] = leg["slope_curve"][1]
        if leg["popt"] is not None:
            save[f"{key}_km_sig_popt"] = np.array(leg["popt"])
        # new keys: the cluster-extent readings, their spread, the diagnostics
        save[f"{key}_cluster_beta_c"] = np.array([rd["beta_lo_high"], rd["beta_hi_low"],
                                                  rd["onset"], rd["n_clusters"]])
        save[f"{key}_cluster_spread"] = np.array([sp["beta_lo_high"], sp["beta_hi_low"], sp["onset"]])
        save[f"{key}_f_high"]         = rd["f_high"]
        save[f"{key}_neck"]           = np.array([r["neck_beta"], r["neck_depth"]])
        save[f"{key}_neck_curve"]     = r["neck_curve"]
        save[f"{key}_worm_rho"]       = np.array([r["worm_rho"], r["step_ratio"]])
        save[f"{key}_settings"]       = np.array([perplexity, r["samples_per_beta"], seed])
        idx = np.random.default_rng(42).choice(len(r["embedding"]),
                                               min(5000, len(r["embedding"])), replace=False)
        save[f"{key}_emb_x"]     = r["embedding"][idx, 0]
        save[f"{key}_emb_y"]     = r["embedding"][idx, 1]
        save[f"{key}_emb_betas"] = r["betas"][idx]
        save[f"{key}_emb_km"]    = leg["labels"][idx]
        save[f"{key}_emb_hdb"]   = rd["labels"][idx]
        store_snapshots(save, key, r)

    fname = result_filename("tsne", tag)
    np.savez(f"{results_dir}/{fname}", **save)
    print(f"\nResults saved to {results_dir}/{fname}")
    return results


def truncation_test(alpha, *, file_template=None, stride=None, perplexity=30,
                    fractions=((0, 1), (0, 2 / 3), (1 / 3, 1))):
    """Refit the embedding on asymmetric beta windows and print every reading.

    A reading that moves with the window midpoint is an artifact of the grid,
    not a property of the data.  ``fractions`` are (start, end) fractions of the
    full beta range."""
    X, betas = load_beta_sweep(alpha, stride=stride, file_template=file_template)
    lo, hi = betas.min(), betas.max()
    print(f"alpha={alpha:.4f}  full grid [{lo:g}, {hi:g}]")
    for f0, f1 in fractions:
        w0, w1 = lo + f0 * (hi - lo), lo + f1 * (hi - lo)
        sel = (betas >= w0 - 1e-9) & (betas <= w1 + 1e-9)
        E = embed(X[sel], perplexity=perplexity)
        rd = cluster_readings(E, betas[sel])
        _, _, neck_b, neck_d = neck_reading(E, betas[sel])
        leg = kmeans_bisection(E, betas[sel])
        print(f"  window [{w0:5.1f}, {w1:5.1f}] mid={(w0 + w1) / 2:5.1f}: NECK={neck_b:5.1f} (depth {neck_d:.2f})"
              f"  clusters={rd['n_clusters']}  beta_lo(high)={rd['beta_lo_high']:5.1f}"
              f"  beta_hi(low)={rd['beta_hi_low']:5.1f}  onset={rd['onset']:5.1f}"
              f"  | legacy km_sig={leg['beta_c']:5.1f} (share {leg['share']:.2f})",
              flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ssh", help="model config to apply (sets the data directory)")
    p.add_argument("--file-template", default=None)
    p.add_argument("--alphas", type=float, nargs="+", default=None)
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=42, help="t-SNE random_state")
    p.add_argument("--stride", type=int, default=None, help="use every N-th row (subsample)")
    p.add_argument("--tag", default=None)
    p.add_argument("--truncation", action="store_true", help="run the truncation test instead")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    _cfg.apply_model(a.model)
    if a.truncation:
        for alpha in (a.alphas or _cfg.ALPHAS):
            truncation_test(alpha, file_template=a.file_template, stride=a.stride,
                            perplexity=a.perplexity)
    else:
        run_all(alphas=a.alphas, stride=a.stride, file_template=a.file_template,
                tag=a.tag, perplexity=a.perplexity, plot=not a.no_plot, model=a.model,
                seed=a.seed)
