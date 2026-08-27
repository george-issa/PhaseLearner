"""Paper figure: KVBS order parameter in the complex plane (oSSH honeycomb, L = 12).

Top row: DQMC samples of the complex Kekule-VBS order parameter rho_KVBS (one
sample per measurement sweep, see fig_kvbs_order_parameter.py for the
definition) are histogrammed in the (Re rho, Im rho) plane at fixed coupling
alpha and a few representative inverse temperatures.  The distribution evolves
from a disordered Gaussian blob centered at the origin, through a ring at beta
near beta_c (emergent U(1): the Z3 anisotropy is weak at L = 12), to a ring
with a clear threefold modulation deep in the KVBS phase -- the three Kekule
domains at 120-degree phase separation.

Bottom row: the real-space counterpart at the same betas -- the integrated
bond-bond correlations chi(r) averaged over the three diagonal bond pairs,
with each panel's off-origin spatial median subtracted (the uniform background
grows with beta; the Kekule signal is the +- modulation around it).  The
period-3 diagonal checkerboard that develops below beta_c is the
sqrt3 x sqrt3 superlattice of cells sharing the same Kekule registry: the
real-space image of the Z3 lobes in the row above.  Color scale shared across
the row; the r = 0 self-term is masked.

Every sample enters the histograms (nothing is thinned); panels pool all four
MPI walkers (~100k samples per beta).  Raw-file reduction takes ~2 min, so
fine-binned histograms are cached in paper/data/; rerun with --rebuild after
adding new runs.  The bond-bond row reads the (small) integrated stats CSVs
directly from data/ssh/sim_data/.

Usage:
    python fig_kvbs_complex_plane.py [--alpha 0.2550] [--betas 4 12.5 20]
                                     [--rebuild]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperstyle as ps

REPO = Path(__file__).resolve().parents[2]
SIM_DATA = REPO / "data" / "ssh" / "sim_data"

# Fine cache binning; the figure rebins by an integer factor (REBIN below).
NBINS_CACHE = 240
RANGE = 3.3  # histogram extent: |Re|, |Im| <= RANGE covers every sample
REBIN = 2    # 240 -> 120 display bins

DEFAULT_ALPHA = 0.2550
DEFAULT_BETAS = (4.0, 12.5, 20.0)  # disordered blob, ring at ~beta_c, Z3 lobes
BETA_C = {0.2400: 17.5, 0.2450: 14.3, 0.2500: 12.5, 0.2550: 12.3}

# Bottom row: integrated position-space bond-bond correlations.
L = 12
DIAG_PAIRS = ((3, 3), (4, 4), (5, 5))  # the three bond orientations
BOND_STATS = "integrated/bond/bond_position_integrated_stats.csv"


def cache_path(alpha):
    return REPO / "paper" / "data" / f"kvbs_complex_plane_a{alpha:.4f}_L12.npz"


def run_re(alpha):
    return re.compile(
        rf"ossh_honeycomb_w0\.5_a{alpha:.4f}_m0\.0_L12_b(?P<beta>[0-9.]+)-\d+$"
    )


def build_cache(alpha):
    """Histogram every (Re, Im) sample of every beta into a fine 2D grid."""
    runs = sorted(d for d in SIM_DATA.iterdir() if run_re(alpha).match(d.name))
    if not runs:
        sys.exit(f"no runs found for alpha={alpha:.4f}")
    edges = np.linspace(-RANGE, RANGE, NBINS_CACHE + 1)
    betas, hists, counts = [], [], []
    for i, run in enumerate(runs, 1):
        beta = float(run_re(alpha).match(run.name)["beta"])
        samples = []
        for f in sorted((run / "kvbs_order_parameter").glob("kvbs_order_parameter_*.txt")):
            re_im = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            if re_im.shape[0]:
                samples.append(re_im)
        if not samples:
            print(f"\nWARNING: no KVBS samples in {run.name}, skipped")
            continue
        re_im = np.vstack(samples)
        h, _, _ = np.histogram2d(re_im[:, 0], re_im[:, 1], bins=(edges, edges))
        betas.append(beta)
        hists.append(h.astype(np.uint32))
        counts.append(re_im.shape[0])
        print(f"\r[{i}/{len(runs)}] b={beta:5.2f} n={re_im.shape[0]}", end="", flush=True)
    print()
    order = np.argsort(betas)
    path = cache_path(alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        alpha=alpha,
        betas=np.array(betas)[order],
        hists=np.array(hists)[order],
        counts=np.array(counts)[order],
        edges=edges,
    )
    print(f"wrote {path}")


def load_cache(alpha, rebuild=False):
    if rebuild or not cache_path(alpha).exists():
        build_cache(alpha)
    return np.load(cache_path(alpha))


def rebin(h, factor):
    n = h.shape[0] // factor
    return h.reshape(n, factor, n, factor).sum(axis=(1, 3))


def load_bond_chi(alpha, beta):
    """chi(r) on the (L, L) cell grid [r2, r1], averaged over the three
    diagonal bond pairs of the integrated position-space stats file.

    Cross pairs are excluded on purpose: their checkerboard is registry-
    shifted by +-2pi/3, so including them would partially cancel the
    pattern; the three diagonal pairs share the same cos(K.r) modulation.
    """
    runs = [d for d in SIM_DATA.iterdir()
            if re.match(rf"ossh_honeycomb_w0\.5_a{alpha:.4f}_m0\.0_L12_"
                        rf"b{beta:.2f}-\d+$", d.name)]
    if not runs:
        sys.exit(f"no run dir for alpha={alpha:.4f} beta={beta:g}")
    d = np.genfromtxt(runs[0] / BOND_STATS, skip_header=1)
    g = np.zeros((L, L))
    for b2, b1 in DIAG_PAIRS:
        sub = d[(d[:, 1] == b2) & (d[:, 2] == b1)]
        g[sub[:, 3].astype(int), sub[:, 4].astype(int)] += sub[:, 5]
    return g / len(DIAG_PAIRS)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS),
                   help="inverse temperatures, one per panel (default: %(default)s)")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild the cached histograms from the raw txt files")
    p.add_argument("--no-guides", dest="guides", action="store_false",
                   help="drop the dotted Z3 guide lines in the last panel")
    args = p.parse_args()

    data = load_cache(args.alpha, rebuild=args.rebuild)
    betas, hists = data["betas"], data["hists"]

    panels = []
    for b in args.betas:
        idx = np.flatnonzero(np.isclose(betas, b))
        if idx.size == 0:
            sys.exit(f"beta={b} not in cache (have: {betas.tolist()})")
        panels.append((b, rebin(hists[idx[0]], REBIN)))

    # Bottom-row data: chi(r) with each panel's off-origin spatial median
    # subtracted -- the uniform background (~3.2 -> 16.8 over these betas)
    # would otherwise dominate a shared scale.  Panels share one symmetric
    # scale, normalized by the global maximum modulation (design survey:
    # Huang et al. npj QM 3, 22 (2018) share one symmetric scale across all
    # real-space correlation panels), so the growth of the checkerboard
    # amplitude is directly comparable and one -1..+1 colorbar serves the
    # row.  The huge r=0 self-term is masked and excluded from the scale.
    bond = []
    for b in args.betas:
        g = load_bond_chi(args.alpha, b)
        dev = g - np.median(np.delete(g.ravel(), 0))
        dev[0, 0] = np.nan
        bond.append(dev)
    bond_max = max(np.nanmax(np.abs(d)) for d in bond)
    bond = [d / bond_max for d in bond]

    # Layout and rendering follow the closest published precedent (same model
    # family): Cohen-Stead et al., PRB 110, 115130 (2024), Fig. 3 -- a single
    # row of square viridis 2D histograms on the colormap's native dark field,
    # beta labels inside the panels, no colorbar (per-panel max normalization,
    # stated in the caption).  PowerNorm keeps the diffuse blob and the sharp
    # lobes simultaneously visible.  The density field is drawn as one filled
    # vector rectangle per bin (pcolormesh, NOT a rasterized image): the whole
    # figure, histogram interiors included, must stay vector.
    from matplotlib.colors import PowerNorm
    import matplotlib.patheffects as pe

    ps.use()
    n = len(panels)
    # 2.92 in: two rows of ~0.958 in square panels + the Re rho label above
    # the top row + the top row's tick-number strip between the rows + the
    # bottom row's tick/axis labels + the slim shared colorbar.  Any less
    # and the square panels become height-limited and shrink off the full
    # column width (check: side margins ~0.5 mm on the 600 dpi render).
    # Rows do not share axes (complex plane vs. cell grid), hence
    # sharex/sharey="row".
    fig, axes = plt.subplots(
        2, n, figsize=(ps.ONE_COLUMN, 2.92), sharex="row", sharey="row",
    )
    # usetex metrics understate the height/depth of math glyphs in the x
    # labels, so constrained layout alone leaves their ink ON the page edge
    # (verified on a 600 dpi pdftoppm render -- the PNG preview hides this).
    # Reserve strips at the bottom (colorbar label descender) AND the top
    # (the Re rho label now sits above the top row) of the layout rect.
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.972))
    bin_edges = np.linspace(-RANGE, RANGE, NBINS_CACHE // REBIN + 1)
    cmap = plt.get_cmap("viridis")
    GUIDE_R = 2.4  # guide radius: crosses the lobes, clears the panel label
    letters = "abcdefgh"

    for k, (ax, (label, (beta, h))) in enumerate(
        zip(axes[0], zip(letters[:n], panels))
    ):
        # edgecolors='face' strokes each cell with its own fill color so
        # adjacent vector rectangles leave no antialiasing hairlines
        ax.pcolormesh(bin_edges, bin_edges, h.T / h.max(), cmap=cmap,
                      norm=PowerNorm(0.45, vmin=0.0, vmax=1.0),
                      edgecolors="face", linewidth=0.05, antialiased=False,
                      rasterized=False)
        ax.set_aspect("equal")
        ax.set_xlim(-RANGE, RANGE)
        ax.set_ylim(-RANGE, RANGE)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        # white inward tick marks stay visible on the dark field; the tick
        # labels outside the axes remain black
        ax.tick_params(which="both", color="white")
        if args.guides and k == n - 1:
            # dotted radial guides at the three Kekule phase angles
            # theta_delta = 0, +-120 deg in the ordered-phase panel only
            # (measured lobe centers: arg<e^{3i theta}>/3 = 0.02 deg pooled
            # over beta t >= 18); no published precedent draws these, so they
            # stay hair-thin and confined to one panel
            for theta in (0.0, 2 * np.pi / 3, -2 * np.pi / 3):
                ax.plot([0, GUIDE_R * np.cos(theta)],
                        [0, GUIDE_R * np.sin(theta)],
                        linestyle=(0, (1, 2.4)), linewidth=0.4,
                        color="white", alpha=0.65)
        ax.text(0.05, 0.93, rf"(\textrm{{{label}}})\ $\beta t = {beta:g}$",
                transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=8.0,
                path_effects=[pe.withStroke(linewidth=0.8,
                                            foreground="#440154")])

    axes[0, 0].set_ylabel(r"$\mathrm{Im}\,\rho_{\mathrm{KVBS}}$")
    # x label sits ON TOP of the top row (user request): the strip between
    # the rows then holds only the top row's tick numbers, pulling the two
    # rows close together
    axes[0, n // 2].set_xlabel(r"$\mathrm{Re}\,\rho_{\mathrm{KVBS}}$",
                               labelpad=3.0)
    axes[0, n // 2].xaxis.set_label_position("top")

    # --- bottom row: real-space integrated bond-bond correlations ----------
    # Same vector-only rendering as the top row (one filled rectangle per
    # cell).  The masked r=0 cell shows the axes face color through the mesh.
    cell_edges = np.arange(L + 1) - 0.5
    bond_cmap = plt.get_cmap("RdBu_r")
    for k, (ax, dev) in enumerate(zip(axes[1], bond)):
        ax.set_facecolor("0.85")
        mesh = ax.pcolormesh(cell_edges, cell_edges,
                             np.ma.masked_invalid(dev),
                             cmap=bond_cmap, vmin=-1.0, vmax=1.0,
                             edgecolors="face", linewidth=0.05,
                             antialiased=False, rasterized=False)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(-0.5, L - 0.5)
        ax.set_xticks([0, 3, 6, 9])   # multiples of the period-3 modulation
        ax.set_yticks([0, 3, 6, 9])
        ax.minorticks_off()           # cell grid: minor ticks are noise
        ax.text(0.05, 0.93, rf"(\textrm{{{letters[n + k]}}})",
                transform=ax.transAxes, ha="left", va="top",
                color="black", fontsize=8.0,
                path_effects=[pe.withStroke(linewidth=1.4,
                                            foreground="white")])
    axes[1, 0].set_ylabel(r"$r_2$")
    # negative labelpad: "r" has no ascender, so the font's ascent
    # whitespace alone leaves the ink closer to the colorbar than to its
    # own tick labels; -1.5 restores the correct visual grouping
    axes[1, n // 2].set_xlabel(r"$r_1$", labelpad=-1.5)

    # One slim shared colorbar for the chi(r) row (design survey: published
    # diverging correlation maps always include one shared bar -- the sign
    # and the zero point are quantitative content).
    cbar = fig.colorbar(mesh, ax=list(axes[1]), location="bottom",
                        fraction=0.06, pad=0.02, shrink=0.5, aspect=30,
                        ticks=[-1, 0, 1])
    cbar.set_label(
        r"$\delta\chi_b(\mathbf{r}) / |\delta\chi_b|_{\mathrm{max}}$",
        labelpad=1.5)
    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(length=2.0, width=0.35)
    # matplotlib rasterizes colorbar solids by default; the whole figure
    # must stay vector (zero image XObjects).  edgecolor="face" strokes
    # each gradient quad with its own fill color -- without it the vector
    # colorbar shows pale antialiasing seams in the PDF (the Agg PNG
    # preview hides them).  The stroke must be ~0.6 pt: a hairline (0.05)
    # is thinner than the rasterizer's blending seam and does not cover it.
    cbar.solids.set_rasterized(False)
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_linewidth(0.6)

    ps.save(fig, "fig_kvbs_complex_plane")


if __name__ == "__main__":
    main()
