"""
Raw-data visualization for SSH phase-detection observables.

Inspect your data BEFORE running any ML.  This module generates:

  Part A — Feature overview (any observable)
      Mean feature value across samples vs beta.
      Feature variance vs beta (the PCA susceptibility signal).

  Part B — Phonon spatial maps
      Quiver plot of the (x,y) phonon displacements on the honeycomb lattice.
      Heatmap of the radial displacement per unit cell.
      Shown for a small set of representative beta values (below/near/above beta_c).

Usage
-----
  python3.11 -m ssh_ml_phase.visualize                                    # default observable
  python3.11 -m ssh_ml_phase.visualize --file-template "radial_disp_..."  # phonon
  python3.11 -m ssh_ml_phase.visualize --alphas 0.240 0.250               # subset
  python3.11 -m ssh_ml_phase.visualize --n-show 4                         # beta snapshots

All figures are saved to results/figures/visualize_<tag>/  and also displayed
interactively if a display is available.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA

from .config import ALPHAS, FSS_DICT, DATA_DIR, RESULTS_DIR, L
from .data import load_beta_sweep, make_tag, ensure_results_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _figures_dir(tag):
    base = os.path.join(RESULTS_DIR, "figures", f"visualize_{tag}" if tag else "visualize")
    os.makedirs(base, exist_ok=True)
    return base


def _beta_sample_indices(u_beta, n_show, fss_bc=None):
    """
    Pick n_show representative beta values spread across the grid,
    always including the point nearest to beta_c if known.
    Returns indices into u_beta.
    """
    idx = np.round(np.linspace(0, len(u_beta) - 1, n_show)).astype(int)
    idx = list(dict.fromkeys(idx))  # deduplicate while preserving order

    if fss_bc is not None:
        bc_idx = int(np.argmin(np.abs(u_beta - fss_bc)))
        if bc_idx not in idx:
            # replace the nearest selected index
            nearest = min(idx, key=lambda i: abs(i - bc_idx))
            idx[idx.index(nearest)] = bc_idx
            idx.sort()
    return idx


def _decode_phonon_row(row, L=12):
    """
    Decode one tau=0 phonon row (576 values) → dict of (L,L) mode arrays.
    Julia memory layout: mode-major, with n1 varying fastest within each mode.
    """
    N = L * L
    modes = row[:4 * N].reshape(4, N)
    return {k: modes[i].reshape(L, L, order='F')
            for i, k in enumerate(("Xa", "Ya", "Xb", "Yb"))}


def _decode_radial_row(row, L=12):
    """
    Decode one radial-displacement row (288 values) → rA, rB each (L,L).

    The flat row interleaves sublattices per unit cell: [A0, B0, A1, B1, ...]
    (column-major (2, N) reshape in the Julia writer — its block-layout
    comment is wrong; see sims/ssh/extract_phonons_edited.py for the verification).
    """
    N = L * L
    return {
        "rA": row[0:2*N:2].reshape(L, L, order='F'),
        "rB": row[1:2*N:2].reshape(L, L, order='F'),
    }


def _is_phonon_tau0(file_template):
    return "phonon_tau0" in file_template


def _is_radial(file_template):
    return "radial_disp" in file_template


# ── Part A: Feature overview ───────────────────────────────────────────────────

def plot_feature_overview(alphas, file_template, beta_min=None, beta_max=None,
                          stride=None, tag=None, fig_dir=None):
    """
    For each alpha: plot (1) mean feature value vs beta and (2) feature variance vs beta.
    The variance curve is the PCA susceptibility signal.
    """
    if fig_dir is None:
        fig_dir = _figures_dir(tag)

    n_alpha = len(alphas)
    fig, axes = plt.subplots(2, n_alpha, figsize=(4 * n_alpha, 6), squeeze=False)
    fig.suptitle(f"Feature overview — {file_template.split('_L')[0]}", fontsize=11)

    for col, alpha in enumerate(alphas):
        try:
            X, betas = load_beta_sweep(alpha, file_template=file_template,
                                       beta_min=beta_min, beta_max=beta_max,
                                       stride=stride)
        except FileNotFoundError:
            for row in range(2):
                axes[row, col].set_visible(False)
            continue

        u_beta = np.unique(betas)
        mean_per_beta = np.array([X[betas == b].mean(axis=0) for b in u_beta])
        var_per_beta  = np.array([X[betas == b].var(axis=0).mean() for b in u_beta])

        fss = FSS_DICT.get(alpha, np.nan)

        # Row 0: mean of all features vs beta (shown as image)
        ax0 = axes[0, col]
        im = ax0.imshow(mean_per_beta.T, aspect="auto",
                        extent=[u_beta[0], u_beta[-1], 0, mean_per_beta.shape[1]],
                        origin="lower", cmap="RdBu_r")
        ax0.set_xlabel("β")
        ax0.set_ylabel("feature index" if col == 0 else "")
        ax0.set_title(f"α={alpha:.4f}\n⟨features⟩ vs β")
        if not np.isnan(fss):
            ax0.axvline(fss, color="lime", lw=1.5, ls="--", label=f"FSS β_c={fss:.1f}")
            ax0.legend(fontsize=7, loc="upper left")
        plt.colorbar(im, ax=ax0, fraction=0.03)

        # Row 1: mean feature variance vs beta
        ax1 = axes[1, col]
        ax1.plot(u_beta, var_per_beta, "o-", ms=3, lw=1.2, color="steelblue")
        if not np.isnan(fss):
            ax1.axvline(fss, color="red", lw=1.5, ls="--", label=f"FSS β_c={fss:.1f}")
            ax1.legend(fontsize=7)
        ax1.set_xlabel("β")
        ax1.set_ylabel("mean feature variance" if col == 0 else "")
        ax1.set_title(f"α={alpha:.4f}\nVar(features) vs β")

    fig.tight_layout()
    fname = os.path.join(fig_dir, "feature_overview.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Part A2: PCA scatter coloured by beta (diagnostic) ────────────────────────

def plot_pca_scatter(alphas, file_template, beta_min=None, beta_max=None,
                     stride=None, tag=None, fig_dir=None):
    """
    Project data onto the first two PCA components and colour points by beta.
    Useful for diagnosing whether the data naturally clusters before any ML.
    """
    if fig_dir is None:
        fig_dir = _figures_dir(tag)

    n_alpha = len(alphas)
    fig, axes = plt.subplots(1, n_alpha, figsize=(4.5 * n_alpha, 4), squeeze=False)
    fig.suptitle(f"PCA(2) coloured by β — {file_template.split('_L')[0]}", fontsize=11)

    for col, alpha in enumerate(alphas):
        ax = axes[0, col]
        try:
            X, betas = load_beta_sweep(alpha, file_template=file_template,
                                       beta_min=beta_min, beta_max=beta_max,
                                       stride=stride)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)

        sc = ax.scatter(Z[:, 0], Z[:, 1], c=betas, cmap="plasma",
                        s=4, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, label="β")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})" if col == 0 else "")
        fss = FSS_DICT.get(alpha, np.nan)
        title = f"α={alpha:.4f}"
        if not np.isnan(fss):
            title += f"  FSS β_c={fss:.1f}"
        ax.set_title(title)

    fig.tight_layout()
    fname = os.path.join(fig_dir, "pca_scatter.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Part B: Phonon spatial maps ────────────────────────────────────────────────

def plot_phonon_maps(alphas, file_template, n_show=4,
                     beta_min=None, beta_max=None, stride=None,
                     tag=None, fig_dir=None):
    """
    For each alpha show n_show beta snapshots as phonon spatial maps.

    For phonon_tau0 files:
      Left panel — quiver plot of (Xa, Ya) displacement for sublattice A.
      Right panel — quiver plot of (Xb, Yb) displacement for sublattice B.

    For radial_disp files:
      Left panel — heatmap of rA (sublattice A radial displacement).
      Right panel — heatmap of rB (sublattice B radial displacement).

    All panels in a single figure: rows = beta snapshots, cols-pairs = alphas.
    """
    if fig_dir is None:
        fig_dir = _figures_dir(tag)

    is_tau0   = _is_phonon_tau0(file_template)
    is_radial = _is_radial(file_template)
    if not (is_tau0 or is_radial):
        print("  plot_phonon_maps: file_template does not look like a phonon file — skipping.")
        return

    panels_per_alpha = 2  # (sublattice A, sublattice B)
    n_alpha = len(alphas)
    ncols = n_alpha * panels_per_alpha
    fig, axes = plt.subplots(n_show, ncols,
                             figsize=(3 * ncols, 3 * n_show),
                             squeeze=False)
    obs_label = "tau=0 phonon field" if is_tau0 else "τ-averaged radial displacement"
    fig.suptitle(f"Phonon spatial maps — {obs_label}", fontsize=11)

    for a_idx, alpha in enumerate(alphas):
        col_base = a_idx * panels_per_alpha
        try:
            X, betas = load_beta_sweep(alpha, file_template=file_template,
                                       beta_min=beta_min, beta_max=beta_max,
                                       stride=stride)
        except FileNotFoundError:
            for row in range(n_show):
                for dc in range(panels_per_alpha):
                    axes[row, col_base + dc].set_visible(False)
            continue

        u_beta = np.unique(betas)
        snap_idxs = _beta_sample_indices(u_beta, n_show, FSS_DICT.get(alpha))

        for row, bidx in enumerate(snap_idxs[:n_show]):
            b = u_beta[bidx]
            # pick one sample from this beta (first available)
            mask = betas == b
            sample = X[mask][0]

            ax_A = axes[row, col_base]
            ax_B = axes[row, col_base + 1]

            if is_tau0:
                d = _decode_phonon_row(sample, L=L)
                _quiver_panel(ax_A, d["Xa"], d["Ya"],
                              title=f"α={alpha:.4f}  β={b:.1f}\nA-site (Xa,Ya)",
                              row=row, col_base=col_base)
                _quiver_panel(ax_B, d["Xb"], d["Yb"],
                              title=f"α={alpha:.4f}  β={b:.1f}\nB-site (Xb,Yb)",
                              row=row, col_base=col_base + 1)
            else:
                d = _decode_radial_row(sample, L=L)
                _heatmap_panel(ax_A, d["rA"],
                               title=f"α={alpha:.4f}  β={b:.1f}\nrA (sublattice A)")
                _heatmap_panel(ax_B, d["rB"],
                               title=f"α={alpha:.4f}  β={b:.1f}\nrB (sublattice B)")

    fig.tight_layout()
    obs_str = "tau0" if is_tau0 else "radial"
    fname = os.path.join(fig_dir, f"phonon_maps_{obs_str}.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def _quiver_panel(ax, Xcomp, Ycomp, title="", row=0, col_base=0):
    """Draw a quiver plot of (Xcomp, Ycomp) on an L×L grid."""
    n1, n2 = Xcomp.shape
    g1, g2 = np.meshgrid(np.arange(n2), np.arange(n1))
    mag = np.sqrt(Xcomp**2 + Ycomp**2)
    ax.quiver(g1, g2, Xcomp, Ycomp, mag,
              cmap="RdYlBu_r", scale=None, scale_units="xy", angles="xy")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def _heatmap_panel(ax, data, title=""):
    """Draw a heatmap of a 2D array."""
    im = ax.imshow(data, cmap="hot", origin="lower", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)


# ── Part B2: Mean phonon field averaged over many samples ─────────────────────

def plot_mean_phonon_field(alphas, file_template, betas_to_show=None,
                           tag=None, fig_dir=None):
    """
    Average all samples at each beta value and show the mean phonon field.
    Averaging over many samples reduces noise and reveals the KVBS pattern clearly.
    """
    if fig_dir is None:
        fig_dir = _figures_dir(tag)

    is_tau0   = _is_phonon_tau0(file_template)
    is_radial = _is_radial(file_template)
    if not (is_tau0 or is_radial):
        print("  plot_mean_phonon_field: not a phonon template — skipping.")
        return

    n_alpha = len(alphas)
    fig, axes = plt.subplots(2, n_alpha, figsize=(4 * n_alpha, 7), squeeze=False)
    obs_str = "tau=0 phonon" if is_tau0 else "radial displacement"
    fig.suptitle(f"Mean {obs_str} — low β (top) vs high β (bottom)", fontsize=11)

    for col, alpha in enumerate(alphas):
        try:
            X, betas = load_beta_sweep(alpha, file_template=file_template)
        except FileNotFoundError:
            for row in range(2):
                axes[row, col].set_visible(False)
            continue

        u_beta = np.unique(betas)
        # Use the lowest and highest beta in the dataset
        b_lo, b_hi = u_beta[0], u_beta[-1]
        fss = FSS_DICT.get(alpha, np.nan)

        for row, b in enumerate([b_lo, b_hi]):
            mask = betas == b
            mean_sample = X[mask].mean(axis=0)
            ax = axes[row, col]

            if is_tau0:
                d = _decode_phonon_row(mean_sample, L=L)
                mag = np.sqrt(d["Xa"]**2 + d["Ya"]**2)
                im = ax.imshow(mag, cmap="hot", origin="lower", aspect="equal")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                d = _decode_radial_row(mean_sample, L=L)
                im = ax.imshow(d["rA"], cmap="hot", origin="lower", aspect="equal")
                plt.colorbar(im, ax=ax, fraction=0.046)

            label = "low β" if row == 0 else "high β"
            bc_note = f"  FSS β_c={fss:.1f}" if not np.isnan(fss) else ""
            ax.set_title(f"α={alpha:.4f}  β={b:.1f} ({label}){bc_note}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fname = os.path.join(fig_dir, f"mean_phonon_{'tau0' if is_tau0 else 'radial'}.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(alphas=None, file_template=None, beta_min=None, beta_max=None,
            stride=None, n_show=4, tag=None):
    from .config import FILE_TEMPLATE as DEFAULT_FILE_TEMPLATE
    if file_template is None:
        file_template = DEFAULT_FILE_TEMPLATE
    if alphas is None:
        alphas = ALPHAS

    resolved_tag = make_tag(beta_min, file_template, tag,
                            beta_max=beta_max, stride=stride)
    fig_dir = _figures_dir(resolved_tag)
    ensure_results_dir()

    obs = file_template.split("_L")[0]
    print(f"\n=== Visualization: {obs} ===")
    print(f"  alphas : {alphas}")
    print(f"  figures: {fig_dir}")

    print("\n[A] Feature overview (mean and variance vs beta) ...")
    plot_feature_overview(alphas, file_template, beta_min=beta_min,
                          beta_max=beta_max, stride=stride,
                          tag=resolved_tag, fig_dir=fig_dir)

    print("[A] PCA(2) scatter coloured by beta ...")
    plot_pca_scatter(alphas, file_template, beta_min=beta_min,
                     beta_max=beta_max, stride=stride,
                     tag=resolved_tag, fig_dir=fig_dir)

    is_phonon = _is_phonon_tau0(file_template) or _is_radial(file_template)
    if is_phonon:
        print(f"[B] Phonon spatial maps ({n_show} beta snapshots per alpha) ...")
        plot_phonon_maps(alphas, file_template, n_show=n_show,
                         beta_min=beta_min, beta_max=beta_max, stride=stride,
                         tag=resolved_tag, fig_dir=fig_dir)

        print("[B] Mean phonon field (low β vs high β) ...")
        plot_mean_phonon_field(alphas, file_template,
                               tag=resolved_tag, fig_dir=fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw observable data")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--file-template", type=str, default=None,
                        help="Filename template (overrides FILE_TEMPLATE in config)")
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--n-show", type=int, default=4,
                        help="Number of beta snapshots for phonon spatial maps")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    run_all(alphas=args.alphas, file_template=args.file_template,
            beta_min=args.beta_min, beta_max=args.beta_max,
            stride=args.stride, n_show=args.n_show, tag=args.tag)


if __name__ == "__main__":
    main()
