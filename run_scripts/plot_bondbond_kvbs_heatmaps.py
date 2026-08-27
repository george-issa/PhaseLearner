#!/usr/bin/env python3
"""
Heatmap visualization of the bond-bond correlation functions and the KVBS
order parameter from the SSH DQMC runs (sID-5).

Data sources (per sim folder data/ssh/sim_data/ossh_..._a{alpha}_..._b{beta}-5/):

  equal-time/bond/bond_position_equal-time_stats.csv     (static,     C(r))
  equal-time/bond/bond_momentum_equal-time_stats.csv     (static,     C(q))
  integrated/bond/bond_position_integrated_stats.csv     (integrated, chi(r))
  integrated/bond/bond_momentum_integrated_stats.csv     (integrated, chi(q))
      864 rows = 6 bond pairs x 144 displacements/momenta.
      Columns: INDEX BOND_ID_2 BOND_ID_1 R_2 R_1 MEAN_R MEAN_I STD
      (K_2 K_1 for momentum).  Pairs: (3,3),(4,3),(4,4),(5,3),(5,4),(5,5)
      over the three NN bonds  3 = A->B (0,0), 4 = A->B (-1,0), 5 = A->B (0,-1).
      Grids decode as g[r2, r1] (g[k2, k1]); position C(r) is real,
      momentum cross pairs are complex.

  kvbs_order_parameter/kvbs_order_parameter_sID-5_pID-{0..3}.txt
      25000 measurement rows per walker: "Re(Phi), Im(Phi)" of the complex
      KVBS order parameter (header spells it KVSB).

KVBS momentum combination (validated against the BondBond012345 CSV mean to
2e-9 relative error, see git history / obsidian notes):

  K(q) = Re[C33+C44+C55] - Re[C43+C53+C54] + sqrt(3)*(Im C43 + Im C54 - Im C53)

Figures saved to data/ssh/figures/, per alpha:

  bondbond_position_equaltime_a{alpha}.png   rows = 6 pairs, cols = 4 betas,
  bondbond_position_integrated_a{alpha}.png  color limits shared per row and
      centered on the off-origin median (the r=0 self-term is excluded -
      it dwarfs the spatial modulation otherwise).

  bondbond_momentum_equaltime_a{alpha}.png   rows = 6 pairs (Re) + KVBS combo,
  bondbond_momentum_integrated_a{alpha}.png  cols = 4 betas, limits shared per
      row excluding the dominant Gamma point (vmax = 99.9th percentile);
      K points (k1,k2)=(4,8),(8,4) circled.

  kvbs_order_parameter_a{alpha}.png          2D log-density histograms of Phi
      in the complex plane at the 4 betas (100k points each: 4 walkers x
      25000), plus <|Phi|> vs beta over the full sweep.

Function -> figure map:

  plot_bond_grid()   all four bondbond_{space}_{kind}_a{alpha}.png figures
                     (main() calls it once per space/kind combination);
                     fed by read_stats() (grids), kvbs_combo() (bottom row
                     of the momentum figures), _row_limits_position() /
                     _row_limits_momentum() (per-row color scales).
  plot_kvbs()        kvbs_order_parameter_a{alpha}.png;
                     fed by read_kvbs() (per-walker Phi time series).
  snapshot_betas()   picks the 4 beta columns used by every figure.
  beta_c()           FSS beta_c shown in titles / sweep panel marker.
  discover_runs()    locates the sim folders all figures read from.

Run:  python3 run_scripts/plot_bondbond_kvbs_heatmaps.py [alpha ...]
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.join(PROJECT_ROOT, "data", "ssh", "sim_data")
FIG_DIR = os.path.join(PROJECT_ROOT, "data", "ssh", "figures")

L = 12
ALPHAS = [0.2400, 0.2425, 0.2450, 0.2500, 0.2550]
PAIRS = [(3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (5, 5)]
K_POINTS = [(4, 8), (8, 4)]  # (k1, k2) of the two inequivalent K points
RUN_RE = re.compile(
    r"ossh_honeycomb_w0\.5_a(?P<alpha>[0-9.]+)_m0\.0_L12_b(?P<beta>[0-9.]+)-5$")

FSS_ALPHA = np.array([0.240, 0.245, 0.250, 0.255])
FSS_BETA_C = np.array([17.5, 14.3, 12.5, 12.3])

STATS_FILES = {
    ("position", "equaltime"): "equal-time/bond/bond_position_equal-time_stats.csv",
    ("momentum", "equaltime"): "equal-time/bond/bond_momentum_equal-time_stats.csv",
    ("position", "integrated"): "integrated/bond/bond_position_integrated_stats.csv",
    ("momentum", "integrated"): "integrated/bond/bond_momentum_integrated_stats.csv",
}


def beta_c(alpha):
    """FSS beta_c: figure suptitles, snapshot targets, sweep-panel marker."""
    return float(np.interp(alpha, FSS_ALPHA, FSS_BETA_C))


def discover_runs():
    """{alpha: {beta: run_dir}} for all sID-5 sim folders (input to all figures)."""
    runs = {}
    for name in sorted(os.listdir(SIM_ROOT)):
        m = RUN_RE.match(name)
        if m:
            runs.setdefault(round(float(m["alpha"]), 4), {})[
                float(m["beta"])] = os.path.join(SIM_ROOT, name)
    return runs


def snapshot_betas(betas, bc):
    """The 4 beta columns of every bondbond_* figure and the 4 histogram
    panels of kvbs_order_parameter_*: min, 0.75*beta_c, beta_c, max."""
    targets = [min(betas), 0.75 * bc, bc, max(betas)]
    picks = [min(betas, key=lambda b: abs(b - t)) for t in targets]
    return list(dict.fromkeys(picks))


def read_stats(run_dir, space, kind):
    """Stats file -> {(b2, b1): complex (L, L) grid indexed [r2/k2, r1/k1]}.
    Provides the panel data for plot_bond_grid (all bondbond_* figures)."""
    path = os.path.join(run_dir, STATS_FILES[(space, kind)])
    d = np.genfromtxt(path, skip_header=1)
    grids = {}
    for b2, b1 in PAIRS:
        sub = d[(d[:, 1] == b2) & (d[:, 2] == b1)]
        g = np.zeros((L, L), dtype=complex)
        g[sub[:, 3].astype(int), sub[:, 4].astype(int)] = sub[:, 5] + 1j * sub[:, 6]
        grids[(b2, b1)] = g
    return grids


def kvbs_combo(grids):
    """K(q) chirality combination: the bottom ("KVBS combo") row of the
    bondbond_momentum_* figures."""
    diag = (grids[(3, 3)] + grids[(4, 4)] + grids[(5, 5)]).real
    cross = grids[(4, 3)] + grids[(5, 3)] + grids[(5, 4)]
    im = np.sqrt(3.0) * (grids[(4, 3)].imag + grids[(5, 4)].imag
                         - grids[(5, 3)].imag)
    return diag - cross.real + im


def read_kvbs(run_dir, n_walkers=4):
    """All walkers' KVBS time series -> complex array (input to plot_kvbs)."""
    vals = []
    for pid in range(n_walkers):
        path = os.path.join(run_dir, "kvbs_order_parameter",
                            f"kvbs_order_parameter_sID-5_pID-{pid}.txt")
        if os.path.exists(path) and os.path.getsize(path) > 64:
            d = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
            vals.append(d[:, 0] + 1j * d[:, 1])
    return np.concatenate(vals) if vals else np.empty(0, dtype=complex)


def _row_limits_position(grids_per_beta, pair):
    """Shared color limits for one pair row of the bondbond_position_*
    figures: r=0 excluded, median-centered."""
    vals = np.concatenate([np.delete(g[pair].real.ravel(), 0)
                           for g in grids_per_beta])
    med = np.median(vals)
    half = max(np.percentile(vals, 99.5) - med, med - np.percentile(vals, 0.5))
    return med - half, med + half, med


def _row_limits_momentum(rows_per_beta):
    """Shared color limits for one row (pair or KVBS combo) of the
    bondbond_momentum_* figures, Gamma point excluded.

    vmax is the 99.9th percentile, not 99.5: a row holds ~4*143 = 572
    pixels, so a 99.5% cap discards the top ~3 values. In the KVBS row
    the only signal pixels are the four K-point values (one per beta
    panel), and 99.5% cut three of them — the scale pinned at the
    smallest K peak and the growth across beta rendered as uniform
    saturation. 99.9% (top ~0.6 values) keeps the scale just below the
    largest K peak so the progression stays visible.
    """
    vals = np.concatenate([np.delete(r.ravel(), 0) for r in rows_per_beta])
    return vals.min(), np.percentile(vals, 99.7)


def plot_bond_grid(alpha, betas, all_grids, space, kind):
    """One bondbond_{space}_{kind}_a{alpha}.png figure (called four times
    per alpha: position/momentum x equaltime/integrated).  Rows = the 6
    bond pairs, plus the kvbs_combo row for momentum; cols = the 4
    snapshot betas; K points circled on momentum panels."""
    bc = beta_c(alpha)
    is_mom = space == "momentum"
    row_keys = PAIRS + (["KVBS"] if is_mom else [])
    nrows, ncols = len(row_keys), len(betas)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.1 * ncols + 1.2, 2.8 * nrows),
                             squeeze=False)
    obs = ("equal-time" if kind == "equaltime" else "integrated")
    fig.suptitle(
        rf"Bond-bond {obs} correlations, {space} space — "
        rf"$\alpha$={alpha:.4f}, FSS $\beta_c$={bc:.1f}"
        "\nbonds: 3 = A$\\to$B (0,0), 4 = A$\\to$B (-1,0), 5 = A$\\to$B (0,-1)"
        + ("; K points circled" if is_mom else
           "; r=(0,0) self-term excluded from color scale"),
        fontsize=11)

    for r, key in enumerate(row_keys):
        if key == "KVBS":
            row_grids = [kvbs_combo(all_grids[b]) for b in betas]
            vmin, vmax = _row_limits_momentum(row_grids)
            cmap, label = "inferno", "KVBS combo"
        elif is_mom:
            row_grids = [all_grids[b][key].real for b in betas]
            vmin, vmax = _row_limits_momentum(row_grids)
            cmap, label = "viridis", f"Re C({key[1]},{key[0]})"
        else:
            row_grids = [all_grids[b][key].real for b in betas]
            vmin, vmax, _ = _row_limits_position(
                [all_grids[b] for b in betas], key)
            cmap, label = "RdBu_r", f"C({key[1]},{key[0]})"

        for c, (b, g) in enumerate(zip(betas, row_grids)):
            ax = axes[r, c]
            im = ax.imshow(g, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if is_mom:
                for k1, k2 in K_POINTS:
                    ax.add_patch(plt.Circle((k1, k2), 0.6, fill=False,
                                            ec="red", lw=1.2))
            if r == 0:
                ax.set_title(rf"$\beta$={b:.1f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(label + "\n$k_2$" if is_mom else
                              label + "\n$r_2$", fontsize=8)
            if r == nrows - 1:
                ax.set_xlabel("$k_1$" if is_mom else "$r_1$")
            ax.set_xticks([0, 4, 8])
            ax.set_yticks([0, 4, 8])
            if c == ncols - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(FIG_DIR, f"bondbond_{space}_{kind}_a{alpha:.4f}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_kvbs(alpha, run_map):
    """The kvbs_order_parameter_a{alpha}.png figure: complex-plane
    log-density histograms of Phi at the 4 snapshot betas, plus the
    <|Phi|> vs beta panel over the full sweep."""
    bc = beta_c(alpha)
    all_series = {b: read_kvbs(run_map[b]) for b in sorted(run_map)}
    empty = [b for b, s in all_series.items() if s.size == 0]
    if empty:
        print(f"  [warn] alpha={alpha:.4f}: no KVBS data for beta={empty}")
        all_series = {b: s for b, s in all_series.items() if s.size}
    betas = snapshot_betas(list(all_series), bc)
    series = {b: all_series[b] for b in betas}
    lim = max(np.percentile(np.abs(s), 99.5) for s in series.values()) * 1.05
    edges = np.linspace(-lim, lim, 101)

    ncols = len(betas) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.6),
                             squeeze=False)
    fig.suptitle(
        rf"KVBS order parameter $\Phi$ — $\alpha$={alpha:.4f}, "
        rf"FSS $\beta_c$={bc:.1f}   "
        "(log-density in the complex plane; 4 walkers x 25000 measurements)",
        fontsize=11)

    for c, b in enumerate(betas):
        ax = axes[0, c]
        s = series[b]
        h = ax.hist2d(s.real, s.imag, bins=[edges, edges], cmap="magma",
                      norm=LogNorm())
        ax.set_aspect("equal")
        ax.axhline(0, color="w", lw=0.4, alpha=0.5)
        ax.axvline(0, color="w", lw=0.4, alpha=0.5)
        ax.set_title(rf"$\beta$={b:.1f}   $\langle|\Phi|\rangle$="
                     rf"{np.abs(s).mean():.3f}", fontsize=9)
        ax.set_xlabel(r"Re $\Phi$")
        if c == 0:
            ax.set_ylabel(r"Im $\Phi$")
        plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)

    # <|Phi|> over the full beta sweep
    ax = axes[0, -1]
    all_b = sorted(all_series)
    mean_abs = [np.abs(all_series[b]).mean() for b in all_b]
    ax.plot(all_b, mean_abs, "o-", ms=3, lw=1.2, color="steelblue")
    ax.axvline(bc, color="red", ls="--", lw=1.2, label=rf"$\beta_c$={bc:.1f}")
    for b in betas:
        ax.axvline(b, color="gray", ls=":", lw=0.7)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle|\Phi|\rangle$")
    ax.set_title("full sweep", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIG_DIR, f"kvbs_order_parameter_a{alpha:.4f}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    """Per alpha: plot_bond_grid x4 (the bondbond_* figures), then
    plot_kvbs (the kvbs_order_parameter_* figure)."""
    alphas = [float(a) for a in sys.argv[1:]] or ALPHAS
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = discover_runs()

    for alpha in alphas:
        run_map = runs[round(alpha, 4)]
        bc = beta_c(alpha)
        # only betas whose run dir has all four stats files
        with_stats = {b: d for b, d in run_map.items()
                      if all(os.path.exists(os.path.join(d, f))
                             for f in STATS_FILES.values())}
        skipped = sorted(set(run_map) - set(with_stats))
        if skipped:
            print(f"  [warn] alpha={alpha:.4f}: no stats for beta={skipped}")
        betas = snapshot_betas(list(with_stats), bc)
        print(f"alpha = {alpha:.4f}  (beta snapshots: "
              f"{', '.join(f'{b:g}' for b in betas)})")

        for space in ("position", "momentum"):
            for kind in ("equaltime", "integrated"):
                grids = {b: read_stats(with_stats[b], space, kind)
                         for b in betas}
                plot_bond_grid(alpha, betas, grids, space, kind)

        plot_kvbs(alpha, run_map)


if __name__ == "__main__":
    main()
