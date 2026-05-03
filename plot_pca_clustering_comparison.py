"""
Compare alternative clustering methods on top of linear PCA, for the
integrated-pair correlator on the attractive-Hubbard pilot.

Linear PCA stays unchanged.  We run it once per observable and then apply
four different clustering schemes to the same (P1, P2) scores:

  baseline   : KMeans(k=2) on raw (P1, P2)            — current canonical PCA
  gmm        : GaussianMixture(k=2, full cov) on (P1, P2)
                — handles elongated / heavy-tailed clusters, soft assignments
  spherical  : L2-normalise (P1, P2) per sample → KMeans on the unit circle
                — kills magnitude entirely, clusters by direction (angle)
  arcsinh    : 1-D KMeans on arcsinh(P1)
                — linearises the exponential PC1 ramp; centroid-pair midpoint
                  lands near β_c instead of in the heavy tail

For each method we compute f_KVBS(β) (fraction of samples at β assigned to
the higher-mean-β cluster), fit a sigmoid for β_c, and read the peak of
df_KVBS/dβ as a second β_c estimator.  All are plotted overlaid; numerical
summary printed to stdout.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit

from ssh_ml_phase import config as _cfg
from ssh_ml_phase.data import load_beta_sweep


plt.rcParams.update({
    "text.usetex":   True,
    "font.family":   "serif",
    "font.serif":    ["Computer Modern Roman"],
    "axes.titlesize":11,
    "axes.labelsize":11,
    "legend.fontsize":8,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Per-observable PDFs land alongside the canonical pipeline plots:
#   results/figures/<model>/pca/<observable>/clustering_comparison.pdf
MODEL_NAME   = "attractive_hubbard"


def _run_dir(obs):
    d = os.path.join(PROJECT_ROOT, "results", "figures", MODEL_NAME,
                     "pca", obs)
    os.makedirs(d, exist_ok=True)
    return d

OBSERVABLES = [
    ("integrated_pair_position",  r"$\chi_{\mathrm{SC}}(\mathbf{x})$"),
    ("integrated_pair_momentum",  r"$\chi_{\mathrm{SC}}(\mathbf{q})$"),
]
ALPHA  = 4.0
BETA_C = 8.5


def sigmoid(beta, beta_c, w):
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


# ── Clustering recipes ────────────────────────────────────────────────────────

def cluster_baseline(P, betas):
    """KMeans on raw (P1, P2) — current canonical PCA clustering."""
    return KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(P[:, :2])


def cluster_gmm(P, betas):
    """GaussianMixture(2) with full covariance on (P1, P2)."""
    return GaussianMixture(n_components=2, covariance_type="full",
                           random_state=42, n_init=10).fit_predict(P[:, :2])


def cluster_spherical(P, betas):
    """Spherical KMeans: L2-normalise each sample's (P1, P2) → KMeans on the
    unit circle.  Drops magnitude entirely, clusters by direction.

    Samples with very small ||(P1, P2)|| have ill-defined direction, but we
    keep them — they're a small minority for this dataset.
    """
    norms = np.linalg.norm(P[:, :2], axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    P_unit = P[:, :2] / norms
    return KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(P_unit)


def cluster_arcsinh(P, betas):
    """1-D KMeans on arcsinh(P1).  arcsinh linearises the exponential PC1
    ramp so the centroid-pair midpoint lands near β_c, not in the tail."""
    return KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(
        np.arcsinh(P[:, :1])
    )


METHODS = [
    ("baseline (KMeans, raw 2-D)",         cluster_baseline,  "#1f77b4"),
    ("GMM (full cov, 2-D)",                cluster_gmm,       "#ff7f0e"),
    ("spherical (KMeans on unit circle)",  cluster_spherical, "#2ca02c"),
    (r"arcsinh($P_1$) (1-D KMeans)",       cluster_arcsinh,   "#d62728"),
]


# ── Cluster → f_KVBS → sigmoid fit ────────────────────────────────────────────

def fkvbs(km_labels, betas, u_beta):
    n0 = (km_labels == 0).sum()
    n1 = (km_labels == 1).sum()
    if n0 == 0 or n1 == 0:
        kvbs = 0
    else:
        m0 = betas[km_labels == 0].mean()
        m1 = betas[km_labels == 1].mean()
        kvbs = 1 if m1 > m0 else 0
    return np.array([np.mean(km_labels[betas == b] == kvbs) for b in u_beta])


def fit_sigmoid(u_beta, fk):
    try:
        popt, pcov = curve_fit(sigmoid, u_beta, fk, p0=[BETA_C, 2.0],
                               bounds=([u_beta[2], 0.1], [u_beta[-3], 20.0]),
                               maxfev=10000)
        return float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        return np.nan, np.nan


# ── Driver ────────────────────────────────────────────────────────────────────

def render_observable_pdf(obs, label, summary):
    """Build a single-observable f_KVBS(β) comparison and save it under
    ``results/figures/<model>/pca/<obs>/clustering_comparison.pdf``."""
    pat = _cfg.OBSERVABLE_DATA_PATTERN
    ddir = os.path.join(_cfg.DATA_DIR, pat["data_subdir"].format(obs=obs))
    ft   = pat["file_template"].format(obs=obs)
    X, betas = load_beta_sweep(ALPHA, data_dir=ddir, file_template=ft)
    Xc = X - X.mean(axis=0)
    P  = PCA(n_components=2).fit_transform(Xc)
    u_beta   = np.unique(betas)
    beta_mid = 0.5 * (u_beta[:-1] + u_beta[1:])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for name, fn, color in METHODS:
        km = fn(P, betas)
        fk = fkvbs(km, betas, u_beta)
        bc_sig, bc_err = fit_sigmoid(u_beta, fk)
        d_fk = np.diff(fk) / np.diff(u_beta)
        bc_deriv = float(beta_mid[np.argmax(d_fk)])

        ax.plot(u_beta, fk, "o-", lw=1.4, ms=3.5, color=color,
                label=(rf"{name}: "
                       rf"$\beta_{{\mathrm{{sig}}}}={bc_sig:.1f}\pm{bc_err:.2f}$, "
                       rf"$\beta_{{d}}={bc_deriv:.1f}$"))
        summary.append((obs, name, bc_sig, bc_err, bc_deriv))

    ax.axvline(BETA_C, color="black", ls=":", lw=1.0)
    ax.text(BETA_C, 0.03, rf" FSS $\beta_c\!=\!{BETA_C}$",
            fontsize=8, va="bottom")
    ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$f_{\mathrm{KVBS}}(\beta)$")
    ax.set_title(f"{label} --- $f_{{\\mathrm{{KVBS}}}}(\\beta)$ "
                 "across clustering schemes", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    fig.suptitle(rf"Linear PCA + alternative clustering --- {obs}  "
                 r"($L=8$, $U=-4$, $\langle n\rangle=0.875$)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(_run_dir(obs), "clustering_comparison.pdf")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    _cfg.apply_model("attractive_hubbard")
    summary = []
    for obs, label in OBSERVABLES:
        render_observable_pdf(obs, label, summary)
    print()

    # Numerical summary
    print(f"{'observable':<28s}  {'method':<38s}  "
          f"{'β_sig':>7s} ± {'err':>5s}   {'β_deriv':>7s}   "
          f"|β−FSS|")
    for obs, name, bc, bc_err, bc_d in summary:
        gap = abs(bc - BETA_C) if not np.isnan(bc) else float("nan")
        print(f"{obs:<28s}  {name:<38s}  "
              f"{bc:>7.2f} ± {bc_err:>5.2f}   {bc_d:>7.2f}   {gap:>6.2f}")


if __name__ == "__main__":
    main()
