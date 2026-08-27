"""Paper figure: KVBS order parameter vs inverse temperature (oSSH honeycomb, L = 12).

The DQMC driver (sims/ssh/oSSH_honeycomb_mpi_execute_edited2.jl, eval_rho_kvbs)
measures the complex Kekule-VBS order parameter once per measurement sweep,

    rho_KVBS = -(1/N) sum_{i in A} e^{-i K.R_i}
                 sum_{delta=1..3} e^{i theta_delta} *
                 e_hat(theta_delta) . (u_bar_{j(i,delta)} - u_bar_i),

with K = K+ - K- the Dirac-momentum difference, theta_delta in {0, -2pi/3, +2pi/3}
the three nearest-neighbor bond angles, and u_bar the imaginary-time-averaged
phonon displacements. Each walker writes one "Re, Im" row per sweep to
    data/ssh/sim_data/<run>/kvbs_order_parameter/kvbs_order_parameter_sID-*_pID-*.txt

Because the three Kekule domains carry phases 120 degrees apart, <rho> averages
to zero; the scalar order parameter plotted here is the modulus average

    S_KVBS(beta) = < |rho_KVBS| > = < sqrt(Re^2 + Im^2) >.

Error bars: samples are binned (bin_size = 100 sweeps, the DQMC bin size) to
remove autocorrelation; the quoted error is the standard error over all bins
of all walkers (4 walkers x 250 bins = 1000 bins per point).

The reduction over ~770 raw files takes a couple of minutes, so the reduced
table is cached in paper/data/; rerun with --rebuild after adding new runs.

Usage:
    python fig_kvbs_order_parameter.py [--alphas 0.24 0.25 ...] [--rebuild]
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
CACHE = REPO / "paper" / "data" / "kvbs_order_parameter_L12_w0.5.csv"

RUN_RE = re.compile(
    r"ossh_honeycomb_w0\.5_a(?P<alpha>[0-9.]+)_m0\.0_L12_b(?P<beta>[0-9.]+)-\d+$"
)
BIN_SIZE = 100  # sweeps per DQMC bin

CACHE_HEADER = (
    "KVBS order parameter vs beta, oSSH honeycomb w=0.5, m=0.0, L=12.\n"
    "m_abs = <|rho_KVBS|> with |rho| = sqrt(Re^2 + Im^2) per measurement sweep;\n"
    f"sem_abs = standard error over bin means (bin_size = {BIN_SIZE} sweeps, all walkers).\n"
    "Source: data/ssh/sim_data/*/kvbs_order_parameter/*.txt "
    "(built by fig_kvbs_order_parameter.py --rebuild)\n"
    "columns: alpha, beta, m_abs, sem_abs, n_bins"
)


def reduce_run(run_dir):
    """Mean and SEM of |rho_KVBS| over all walkers of one (alpha, beta) run."""
    bin_means = []
    for f in sorted((run_dir / "kvbs_order_parameter").glob("kvbs_order_parameter_*.txt")):
        re_im = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
        n_bins = re_im.shape[0] // BIN_SIZE
        if n_bins == 0:  # header-only file from a crashed/incomplete walker
            continue
        mod = np.hypot(re_im[:, 0], re_im[:, 1])
        bin_means.append(mod[: n_bins * BIN_SIZE].reshape(n_bins, BIN_SIZE).mean(axis=1))
    if not bin_means:
        return None
    bins = np.concatenate(bin_means)
    return bins.mean(), bins.std(ddof=1) / np.sqrt(len(bins)), len(bins)


def build_cache():
    rows = []
    runs = sorted(d for d in SIM_DATA.iterdir() if RUN_RE.match(d.name))
    for i, run in enumerate(runs, 1):
        m = RUN_RE.match(run.name)
        alpha, beta = float(m["alpha"]), float(m["beta"])
        reduced = reduce_run(run)
        if reduced is None:
            print(f"\nWARNING: no KVBS samples in {run.name}, skipped")
            continue
        rows.append((alpha, beta, *reduced))
        print(f"\r[{i}/{len(runs)}] a={alpha:.4f} b={beta:5.2f}", end="", flush=True)
    print()
    rows = np.array(sorted(rows))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(CACHE, rows, fmt="%.4f %.2f %.8e %.8e %d", header=CACHE_HEADER)
    return rows


def load_table(rebuild=False):
    if rebuild or not CACHE.exists():
        return build_cache()
    return np.loadtxt(CACHE, ndmin=2)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="subset of alpha values to plot (default: all in cache)")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild the reduced-data cache from the raw txt files")
    p.add_argument("--mark-betac", action="store_true",
                   help="mark the FSS reference beta_c per alpha as small "
                        "colored ticks on the bottom axis")
    args = p.parse_args()

    table = load_table(rebuild=args.rebuild)
    alphas = sorted(set(table[:, 0]))
    if args.alphas is not None:
        wanted = set(round(a, 4) for a in args.alphas)
        alphas = [a for a in alphas if round(a, 4) in wanted]
        if not alphas:
            sys.exit(f"no cached data for alphas {sorted(wanted)}")

    ps.use()
    fig, ax = plt.subplots(figsize=ps.figsize())
    colors = ps.sequential(len(alphas), cmap="viridis")
    fallback = iter(m for m in ps.MARKERS if m not in ps.ALPHA_MARKER.values())

    for alpha, color in zip(alphas, colors):
        marker = ps.ALPHA_MARKER.get(round(alpha, 4)) or next(fallback)
        sel = table[table[:, 0] == alpha]
        sel = sel[np.argsort(sel[:, 1])]
        ax.errorbar(
            sel[:, 1], sel[:, 2], yerr=sel[:, 3],
            color=color, marker=marker, markersize=2.2, linewidth=0.7,
            markerfacecolor="none", markeredgecolor=color,
            label=rf"${alpha:.4f}$",
        )

    if args.mark_betac:
        # FSS reference beta_c (obsidian/03_Source_Map.md); none yet for 0.2425.
        beta_c = {0.2400: 17.5, 0.2450: 14.3, 0.2500: 12.5, 0.2550: 12.3}
        import matplotlib.transforms as mtransforms
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for alpha, color in zip(alphas, colors):
            bc = beta_c.get(round(alpha, 4))
            if bc is not None:
                ax.plot([bc, bc], [0.0, 0.035], transform=tr,
                        color=color, linewidth=0.9, solid_capstyle="butt")

    ax.set_xlabel(r"$\beta t$")
    ax.set_ylabel(r"$S_{\mathrm{KVBS}}$")
    ax.text(0.97, 0.05, r"$L = 12$", transform=ax.transAxes,
            ha="right", va="bottom")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="upper left",
                    title=r"$\alpha$", alignment="left")  # order matches stacking
    # center the title over the handle/marker column (handlelength 1.5 x 8 pt)
    leg.get_title().set_position((4.0, 0))
    # data markers are transparent; white-fill the legend keys only, so the
    # key line/error bar don't cross the shape and the decoder stays legible
    import matplotlib.lines as mlines
    for artist in leg.findobj(mlines.Line2D):
        if artist.get_marker() not in ("", "None", None):
            artist.set_markerfacecolor("white")

    ps.save(fig, "fig_kvbs_order_parameter")


if __name__ == "__main__":
    main()
