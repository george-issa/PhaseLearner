"""Shared matplotlib style for all PhaseLearner paper figures (PRX / REVTeX 4.2).

Every paper figure script imports this module first:

    import paperstyle as ps
    ps.use()
    fig, ax = plt.subplots(figsize=ps.figsize())            # single column (default)
    fig, ax = plt.subplots(figsize=ps.figsize('double'))    # full text width
    ...
    ps.save(fig, "fig_kvbs_order_parameter")                # -> paper/figures/*.pdf (+ .png preview)

Conventions (standing rules — every paper figure must satisfy all of these)
---------------------------------------------------------------------------
* Default figure width is ONE column of the two-column PRX layout (86 mm).
  Use the double width only when a figure genuinely cannot fit one column.
* All text is rendered with real LaTeX in classic Computer Modern / Latin
  Modern (usetex + lmodern) — the default Overleaf/VS Code LaTeX look the
  author requires; light strokes, tall letterforms. Do NOT switch to STIX
  or Times packages.
* Axis limits frame the data tightly — no unnecessary white space.
* Curves are thin; frames, ticks, and marker edges are extra thin.
* Every data point that was run is shown — markevery/errorevery are BANNED.
* Each member of a curve family gets its own marker shape (MARKERS below)
  in addition to its color.
* The KVBS order parameter is always denoted $S_{\\mathrm{KVBS}}$.
* Okabe-Ito colorblind-safe palette for categorical series; sequential()
  for ordered families of curves (e.g. an alpha or L sweep).
* Vector PDF output; a PNG preview is written alongside for inspection.
* Per figure, TWO agents are always used: (1) a design-research agent that
  surveys comparable figures in published PRX papers for ideas, and (2) a
  reviewer agent that checks clarity, overlaps, and style consistency —
  iterate until it approves.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Geometry (inches). APS/PRX reprint layout: column = 86 mm, text = 178 mm.
# ---------------------------------------------------------------------------
ONE_COLUMN = 3.375
TWO_COLUMN = 7.0
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# Okabe-Ito palette (colorblind safe), black excluded.
OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
)

# Marker sequence for families of curves: every member of a family (e.g. each
# alpha in a sweep) gets its own marker shape, in this order. NEVER thin
# markers out (no markevery) — every data point that was run must be visible.
MARKERS = ("o", "s", "D", "^", "v", "p", "h")

# Locked alpha -> marker map, reused IDENTICALLY in every figure that shows
# these couplings (PRX convention: a symbol keeps its meaning across figures).
ALPHA_MARKER = {
    0.2400: "o",
    0.2425: "s",
    0.2450: "D",
    0.2500: "^",
    0.2550: "v",
}

_RC = {
    # --- fonts: real LaTeX (required for all paper figures) -----------------
    # Classic Computer Modern / Latin Modern — the default LaTeX face you get
    # in Overleaf/VS Code. The author wants exactly this look in figure math
    # (STIX Two and newtx Times were both tried and rejected as too heavy).
    "text.usetex": True,
    "text.latex.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
    ]),
    "font.family": "serif",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    # --- axes / lines: thin curves, extra-thin frames and marker edges ------
    "axes.linewidth": 0.35,
    "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    "lines.linewidth": 0.7,
    "lines.markersize": 2.4,
    "lines.markeredgewidth": 0.35,
    "errorbar.capsize": 1.2,
    # --- limits: frame the data tightly, no dead white space -----------------
    "axes.xmargin": 0.02,
    "axes.ymargin": 0.04,
    "axes.autolimit_mode": "data",
    # --- ticks: inward, all four sides, minors on ---------------------------
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.35,
    "ytick.major.width": 0.35,
    "xtick.minor.width": 0.25,
    "ytick.minor.width": 0.25,
    # --- legend --------------------------------------------------------------
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.3,
    # --- layout / output -----------------------------------------------------
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use():
    """Activate the paper style (call once, before creating any figure)."""
    mpl.rcParams.update(_RC)


def figsize(width="single", aspect=GOLDEN):
    """(width, height) in inches.

    width  : 'single' (default, one PRX column), 'double', or a float in inches.
    aspect : width/height ratio; golden ratio by default.
    """
    w = {"single": ONE_COLUMN, "double": TWO_COLUMN}.get(width, width)
    return (w, w / aspect)


def sequential(n, cmap="viridis", lo=0.0, hi=0.82):
    """n perceptually-ordered colors for an ordered family of curves."""
    return [plt.get_cmap(cmap)(v) for v in np.linspace(lo, hi, n)]


def panel_label(ax, text, x=0.04, y=0.96, **kwargs):
    """Place a PRX-style panel label, e.g. '(a)', inside the axes corner."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", **kwargs)


def save(fig, name, png_preview=True):
    """Save into paper/figures/ as vector PDF (+ optional PNG preview)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    if png_preview:
        fig.savefig(FIGURES_DIR / f"{name}.png")
    print(f"wrote {pdf}")
