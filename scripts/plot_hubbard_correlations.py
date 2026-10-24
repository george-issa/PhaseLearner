"""
Quick physics inspection of the attractive-Hubbard correlation data
(U=-4, n=0.875, L=8, β∈[2,14] step 0.5).

Loops the eight per-beta CSV directories under data/csv/, averages over
DQMC bins, and writes four figures into figures/hubbard_pilot/:

  correlations_vs_beta.pdf            2×4 panels, each observable vs β
                                       — q/x=(0,0), corner, max
  spatial_profiles.pdf                2×4 panels, feature-index profile
                                       at three representative β
  heatmaps_position_observables.pdf   4 pages (one per real-space obs);
                                       grid of 8×8 lattice heatmaps,
                                       one heatmap per β
  heatmaps_momentum_observables.pdf   4 pages (one per momentum obs);
                                       grid of 8×8 BZ heatmaps,
                                       one heatmap per β

Reference T_c at U=-4, n=0.875: T_c≈0.118 → β_c≈8.5  (Paiva et al. 2004).
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Render all text with LaTeX. Requires a working TeX install on PATH
# (latex, dvipng). The repo already builds .tex documents, so this is fine.
plt.rcParams.update({
    "text.usetex":       True,
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman"],
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Raw-data inspection plots live alongside the data they describe:
# data/<model>/figures/  — sibling of  data/<model>/sim_data/  and  data/<model>/csv/.
OUTDIR = os.path.join(PROJECT_ROOT, "data", "hubbard_square", "figures")
os.makedirs(OUTDIR, exist_ok=True)

L = 8
BETA_C = 8.5  # literature reference for U=-4, n=0.875

#  (name, axis-title (LaTeX), log_y, momentum_space?)
OBSERVABLES = [
    ("density_position",            r"$S^{N}(\mathbf{x})$ density--density (real)",                            False, False),
    ("density_momentum",            r"$N(\mathbf{q})$ charge structure factor",                                False, True),
    ("pair_position",               r"$P(\mathbf{x})$ pair correlator (real)",                                 False, False),
    ("pair_momentum",               r"$P(\mathbf{q})$ pair structure factor",                                  False, True),
    ("integrated_density_position", r"$\chi^{N}(\mathbf{x})$ $\tau$-integrated density (real)",                True,  False),
    ("integrated_density_momentum", r"$\chi^{N}(\mathbf{q})$ $\tau$-integrated charge ($q$-space)",            True,  True),
    ("integrated_pair_position",    r"$\chi_{\mathrm{SC}}(\mathbf{x})$ $\tau$-integrated pair (real)",        True,  False),
    ("integrated_pair_momentum",    r"$\chi_{\mathrm{SC}}(\mathbf{q})$ $\tau$-integrated pair ($q$-space)",   True,  True),
]

DIR_TEMPLATE  = "data/hubbard_square/csv/{name}_L8_n0.875_U4.0000-1_csv"
FILE_TEMPLATE = "{name}_L8_n0.875_b{beta:.2f}_U4.0000-1.csv"
BETA_RE = re.compile(r"_b([0-9]+\.[0-9]+)_U")


def load_observable(name):
    """Return betas (sorted) and arrays of shape (n_beta, n_features) for mean & std."""
    d = os.path.join(PROJECT_ROOT, DIR_TEMPLATE.format(name=name))
    rows = []
    for fpath in glob.glob(os.path.join(d, "*_b*_U*.csv")):
        m = BETA_RE.search(os.path.basename(fpath))
        if not m:
            continue
        beta = float(m.group(1))
        data = np.loadtxt(fpath, delimiter=",")
        # last column is beta (constant per file); features are 0..-2
        feats = data[:, :-1]
        rows.append((beta, feats.mean(axis=0), feats.std(axis=0), feats.shape[0]))
    rows.sort(key=lambda r: r[0])
    betas = np.array([r[0] for r in rows])
    means = np.stack([r[1] for r in rows])      # (n_beta, n_features)
    stds  = np.stack([r[2] for r in rows])
    nbins = np.array([r[3] for r in rows])
    return betas, means, stds, nbins


def idx_2d(ix, iy, L=L):
    """Row-major linear index for an (ix, iy) grid point on an L×L lattice."""
    return ix * L + iy


def panel_vs_beta(ax, name, label, log_y, momentum):
    betas, means, stds, nbins = load_observable(name)

    # Key indices: q=(0,0) / x=(0,0)  and  q=(π,π) / x=(L/2,L/2)
    i00    = idx_2d(0, 0)
    icornr = idx_2d(L // 2, L // 2)

    # standard error of the mean across DQMC bins
    sem = stds / np.sqrt(np.maximum(nbins, 1))[:, None]

    if momentum:
        l_origin = r"$\mathbf{q}=(0,0)$"
        l_corner = r"$\mathbf{q}=(\pi,\pi)$"
    else:
        l_origin = r"$\mathbf{x}=(0,0)$"
        l_corner = r"$\mathbf{x}=(L/2,L/2)$"

    ax.errorbar(betas, means[:, i00],    yerr=sem[:, i00],
                fmt="o-", ms=4, lw=1.4, label=l_origin, color="#1f77b4")
    ax.errorbar(betas, means[:, icornr], yerr=sem[:, icornr],
                fmt="s-", ms=4, lw=1.4, label=l_corner, color="#d62728")
    ax.plot(betas, means.max(axis=1), "k--", lw=1.0, alpha=0.7,
            label=r"max over features")

    ax.axvline(BETA_C, color="gray", ls=":", lw=1.0)
    ax.text(BETA_C, 0.98, rf"\ $\beta_c\approx{BETA_C}$",
            color="gray", fontsize=8,
            va="top", ha="left", transform=ax.get_xaxis_transform())

    ax.set_title(label, fontsize=10)
    ax.set_xlabel(r"$\beta$")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return betas, means


def panel_spatial(ax, name, label, log_y, momentum):
    betas, means, _, _ = load_observable(name)
    # Pick three representative betas: smallest, near β_c, largest.
    targets = [betas.min(), BETA_C, betas.max()]
    chosen = [betas[np.argmin(np.abs(betas - t))] for t in targets]
    cmap = plt.get_cmap("viridis")
    for k, b in enumerate(chosen):
        i = int(np.where(betas == b)[0][0])
        ax.plot(np.arange(means.shape[1]), means[i], "-",
                lw=1.3, color=cmap(k / max(len(chosen) - 1, 1)),
                label=rf"$\beta = {b:g}$")
    ax.set_title(label, fontsize=10)
    if momentum:
        ax.set_xlabel(r"feature index $i_x L + i_y$ ($\mathbf{q}$-space)")
    else:
        ax.set_xlabel(r"feature index $i_x L + i_y$ ($\mathbf{x}$-space)")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def heatmap_grid_page(pdf, name, label, momentum, log_scale,
                      exclude_origin=False):
    """One PDF page: a grid of 8×8 lattice heatmaps, one per β.

    Each subplot is the mean correlator value reshaped to (L, L); the
    panel title is the corresponding β. All subplots share a single
    color scale (linear, or log if `log_scale`) so magnitudes can be
    compared visually across temperatures. Position panels show
    (i_x, i_y) on the lattice; momentum panels show
    q = (2π/L)(i_x, i_y) (q=0 at top-left, q=(π,π) at index (L/2,L/2)).

    If `exclude_origin`, the (0,0) cell — q=(0,0) for momentum, or x=(0,0)
    for position — is masked and excluded from the shared color scale, so
    finite-q (or finite-x) structure is not drowned out by the dominant
    DC component.
    """
    from matplotlib.colors import LogNorm

    betas, means, _, _ = load_observable(name)
    n_beta = len(betas)
    grids = means.reshape(n_beta, L, L).astype(float)   # row-major reshape

    if exclude_origin:
        grids = grids.copy()
        grids[:, 0, 0] = np.nan
        vmin = float(np.nanmin(grids))
        vmax = float(np.nanmax(grids))
    else:
        vmin = float(grids.min())
        vmax = float(grids.max())

    ncols = 5
    nrows = int(np.ceil(n_beta / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.2 * ncols + 1.2, 2.2 * nrows + 0.6),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    if log_scale and vmin > 0:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        kwargs = dict(norm=norm)
    else:
        kwargs = dict(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")
    last_im = None
    for k, b in enumerate(betas):
        ax = axes_flat[k]
        last_im = ax.imshow(grids[k], origin="upper", cmap=cmap,
                            interpolation="nearest", **kwargs)
        ax.set_title(rf"$\beta = {b:g}$", fontsize=8, pad=2)
        ax.set_xticks([0, L // 2, L - 1])
        ax.set_yticks([0, L // 2, L - 1])
        ax.tick_params(labelsize=6, length=2)

    # Hide any unused cells in the last row
    for k in range(n_beta, len(axes_flat)):
        axes_flat[k].axis("off")

    if momentum:
        ax_label_x = r"$i_y$ \quad ($q_y = 2\pi i_y/L$)"
        ax_label_y = r"$i_x$ \quad ($q_x = 2\pi i_x/L$)"
        title_suffix = r" --- $L\times L$ heatmap on the BZ at each $\beta$"
        if exclude_origin:
            title_suffix += r" \ ($\mathbf{q}=(0,0)$ masked)"
    else:
        ax_label_x = r"$i_y$"
        ax_label_y = r"$i_x$"
        title_suffix = r" --- $L\times L$ heatmap on the lattice at each $\beta$"
        if exclude_origin:
            title_suffix += r" \ ($\mathbf{x}=(0,0)$ masked)"

    fig.suptitle(label + title_suffix, fontsize=12)
    fig.supxlabel(ax_label_x, fontsize=10)
    fig.supylabel(ax_label_y, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.07,
                        hspace=0.35, wspace=0.15)
    cax = fig.add_axes([0.90, 0.10, 0.02, 0.78])
    fig.colorbar(last_im, cax=cax)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    # --- Figure 1: each observable vs β -------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, (name, label, log_y, mom) in zip(axes.flat, OBSERVABLES):
        panel_vs_beta(ax, name, label, log_y, mom)
    fig.suptitle(r"Attractive Hubbard ($U=-4$, $\langle n\rangle=0.875$, "
                 r"$L=8$) --- correlation observables vs $\beta$",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(OUTDIR, "correlations_vs_beta.pdf")
    fig.savefig(out1)
    print(f"wrote {out1}")

    # --- Figure 2: spatial / momentum profile at representative β -----------
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, (name, label, log_y, mom) in zip(axes.flat, OBSERVABLES):
        panel_spatial(ax, name, label, log_y, mom)
    fig.suptitle(r"Spatial / momentum profiles at three representative $\beta$",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(OUTDIR, "spatial_profiles.pdf")
    fig.savefig(out2)
    print(f"wrote {out2}")

    # --- Figures 3 & 4: per-β 8×8 heatmaps, one observable per page ---------
    pos_obs = [o for o in OBSERVABLES if not o[3]]
    mom_obs = [o for o in OBSERVABLES if     o[3]]

    out3 = os.path.join(OUTDIR, "heatmaps_position_observables.pdf")
    with PdfPages(out3) as pdf:
        for name, label, log_scale, _ in pos_obs:
            heatmap_grid_page(pdf, name, label,
                              momentum=False, log_scale=log_scale)
    print(f"wrote {out3}")

    out4 = os.path.join(OUTDIR, "heatmaps_momentum_observables.pdf")
    with PdfPages(out4) as pdf:
        for name, label, log_scale, _ in mom_obs:
            heatmap_grid_page(pdf, name, label,
                              momentum=True, log_scale=log_scale)
    print(f"wrote {out4}")

    # --- Figure 5: charge (density) momentum heatmaps with q=(0,0) masked ---
    # The q=0 component is the squared total charge and dominates the color
    # scale; masking it lets the finite-q structure (in particular any
    # build-up near q=(π,π) for CDW tendencies) become visible.
    charge_mom_obs = [o for o in OBSERVABLES
                      if o[3] and o[0].endswith("density_momentum")]
    out5 = os.path.join(OUTDIR, "heatmaps_charge_momentum_no_q0.pdf")
    with PdfPages(out5) as pdf:
        for name, label, log_scale, _ in charge_mom_obs:
            heatmap_grid_page(pdf, name, label,
                              momentum=True, log_scale=log_scale,
                              exclude_origin=True)
    print(f"wrote {out5}")


if __name__ == "__main__":
    main()
