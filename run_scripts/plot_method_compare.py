#!/usr/bin/env python3
"""
Heatmap of PCA / UMAP / t-SNE agreement with FSS, per oSSH observable.

Reads the .npz files written by run_methods_ssh.py and renders one cell per
(observable, method) coloured by relRMSE — the RMS of (beta_c - FSS)/FSS over
the four FSS-anchored alphas.  Lighter is better.

Two panels, one per shared estimator (km_sig = sigmoid midpoint of the
ordered-cluster fraction, km_deriv = steepest-slope point of the same curve),
because which method wins depends on which of the two you read.

The best method in each row is outlined.  A trailing ``*`` marks an estimator
whose beta_c rises by more than 1.0 between consecutive alphas — the physical
boundary falls with alpha, so a large rise means the number is not tracking
the transition however good its RMSE looks.

Run:  python run_scripts/plot_method_compare.py
      python run_scripts/plot_method_compare.py --sid method_compare
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]
FSS = {0.240: 17.5, 0.245: 14.3, 0.250: 12.5, 0.255: 12.3}

METHODS = [("pca", "PCA"), ("umap", "UMAP"), ("tsne", "t-SNE")]
ESTIMATORS = [("km_sig", "sigmoid midpoint  (km_sig)"),
              ("km_deriv", "steepest slope  (km_deriv)")]
INDEX = {                       # position in each method's saved beta_c array
    "pca":  {"km_sig": 2, "km_deriv": 4},
    "umap": {"km_sig": 0, "km_deriv": 2},
    "tsne": {"km_sig": 0, "km_deriv": 2},
}

BOND_PAIRS = ["33", "34", "35", "44", "45", "55"]
GREENS_PAIRS = ["11", "12", "22"]
SPACES = ["position", "momentum"]

# Sequential blue ramp, steps 100 -> 700 (dataviz reference palette).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
WIN_COLOR = "#eb6834"           # categorical slot 2 — distinct from the ramp
RISE_LIMIT = 1.0


def row_groups():
    """Observable rows, grouped by family; returns [(group title, [obs, ...]), ...]."""
    integ = [f"integrated_bond{ab}_{s}" for ab in BOND_PAIRS for s in SPACES]
    integ.append("integrated_BondBond012345_momentum")
    static = [f"bond{ab}_{s}" for ab in BOND_PAIRS for s in SPACES]
    static.append("BondBond012345_momentum")
    greens = [f"greens{ab}_{s}" for ab in GREENS_PAIRS for s in SPACES]
    return [
        ("Bond-bond, imaginary-time integrated (susceptibility)", integ),
        ("Bond-bond, equal time", static),
        ("Green's function, equal time", greens),
        ("Phonon", ["phonon_tau0", "radial_disp"]),
    ]


def score(npz_dir, method, obs, estimator):
    """(relRMSE, max_rise) for one cell; (nan, nan) if the file is missing."""
    path = os.path.join(npz_dir, f"{method}_results_{obs}_sid5.npz")
    if not os.path.exists(path):
        return np.nan, np.nan
    d = np.load(path, allow_pickle=True)
    idx = INDEX[method][estimator]
    vals = np.array([float(d[f"a{a:.4f}_beta_c"][idx])
                     if f"a{a:.4f}_beta_c" in d.files else np.nan
                     for a in ALPHAS])
    err = [(vals[i] - FSS[a]) / FSS[a]
           for i, a in enumerate(ALPHAS) if a in FSS and np.isfinite(vals[i])]
    relrmse = np.sqrt(np.mean(np.square(err))) if len(err) == len(FSS) else np.nan
    finite = vals[np.isfinite(vals)]
    rise = float(max(0.0, np.max(np.diff(finite)))) if len(finite) > 1 else np.nan
    return relrmse, rise


def pretty(obs):
    """Shorter row label — the family is already in the group heading."""
    return (obs.replace("integrated_", "")
               .replace("BondBond012345", "KVBS combo")
               .replace("_", "  "))


def build_rows(npz_dir, estimator, groups):
    """Rows for one panel: a NaN spacer row per family, then its observables.

    The spacer carries the family name as its tick label, which keeps the
    headings out of the cells entirely.
    """
    labels, values, rises, is_heading = [], [], [], []
    for title, obs_list in groups:
        labels.append(title)
        values.append([np.nan] * len(METHODS))
        rises.append([np.nan] * len(METHODS))
        is_heading.append(True)
        for obs in obs_list:
            row = [score(npz_dir, m, obs, estimator) for m, _ in METHODS]
            labels.append(pretty(obs))
            values.append([r[0] for r in row])
            rises.append([r[1] for r in row])
            is_heading.append(False)
    return labels, np.array(values), np.array(rises), is_heading


def draw_panel(ax, npz_dir, estimator, groups, cmap, norm, show_ylabels):
    labels, values, rises, is_heading = build_rows(npz_dir, estimator, groups)

    cmap = cmap.copy()
    cmap.set_bad("#fcfcfb")                      # spacer rows read as surface
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([n for _, n in METHODS], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if show_ylabels else [""] * len(labels), fontsize=7.2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    if show_ylabels:
        for tick, heading in zip(ax.get_yticklabels(), is_heading):
            if heading:
                tick.set_fontsize(8.0)
                tick.set_style("italic")
                tick.set_color("#52514e")

    for i in range(len(labels)):
        if is_heading[i] or not np.isfinite(values[i]).any():
            continue
        best = int(np.nanargmin(values[i]))
        for j in range(len(METHODS)):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            mark = "*" if np.isfinite(rises[i, j]) and rises[i, j] > RISE_LIMIT else ""
            # Dark cells need light text; the ramp crosses over around 0.20.
            ink = "#ffffff" if norm(v) > 0.55 else "#0b0b0b"
            ax.text(j, i, f"{v:.3f}{mark}", ha="center", va="center",
                    fontsize=7.2, color=ink,
                    fontweight="bold" if j == best else "normal")
            if j == best:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=WIN_COLOR, linewidth=1.6,
                                       zorder=3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sid", default="method_compare")
    ap.add_argument("--out", default="results/figures/method_compare_heatmap")
    args = ap.parse_args()

    npz_dir = os.path.join(ROOT, "results", args.sid, "npz")
    if not os.path.isdir(npz_dir):
        sys.exit(f"no npz dir at {npz_dir}")

    groups = row_groups()
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=0.55)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 9.8))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (est, est_label) in zip(axes, ESTIMATORS):
        ax.set_facecolor("#fcfcfb")
        draw_panel(ax, npz_dir, est, groups, cmap, norm,
                   show_ylabels=(ax is axes[0]))
        ax.set_title(est_label, fontsize=9.5, pad=20, color="#0b0b0b")

    fig.suptitle("Which method recovers the oSSH $\\beta_c$ best, per observable",
                 fontsize=11.5, x=0.5, y=0.985, color="#0b0b0b")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, orientation="horizontal",
                        fraction=0.030, pad=0.045, aspect=46)
    cbar.set_label("relative RMSE vs FSS $\\beta_c$   (lower = better)",
                   fontsize=8.5, color="#0b0b0b")
    cbar.ax.tick_params(labelsize=7.5, length=2)
    cbar.outline.set_visible(False)

    fig.text(0.5, 0.032,
             "orange outline = best method for that observable      "
             f"*  = $\\beta_c$ rises by more than {RISE_LIMIT:.0f} between "
             "consecutive $\\alpha$ (not tracking the boundary)",
             ha="center", fontsize=7.4, color="#52514e")

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
