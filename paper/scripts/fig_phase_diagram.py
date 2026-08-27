#!/usr/bin/env python3
r"""Phase diagram of the optical-SSH model: critical temperature
T_c vs electron-phonon coupling alpha (L = 12 honeycomb, omega = 0.5).

T_c = 1 / beta_c (beta = beta*t is dimensionless, so T_c is in units of the
hopping t).  The ordered Kekule valence-bond-solid (KVBS) phase lives at LOW T
(high beta), the disordered semimetal at HIGH T -- so on this T_c-alpha diagram
the KVBS region sits BELOW the phase boundary and the semimetal ABOVE it, and
the boundary RISES with alpha (beta_c falls with alpha, so T_c grows).
Two series are overlaid:

  * FSS        -- finite-size-scaling beta_c (sohan_phase_diagram.pdf); the
                  established phase boundary / ground truth.  Drawn as a smooth
                  guide-to-the-eye that separates the two coloured regions.
  * PCA / k-means -- unsupervised estimate from the integrated single
                  bond-bond correlator B_33: PCA scores clustered with k=2
                  k-means, beta_c = sigmoid midpoint of the ordered fraction
                  (the ``km_sig`` estimator).  Validated against FSS.

The two phases are cued by small honeycomb cartoons: a uniform lattice for the
semimetal and a Kekule (alternating strong-bond) lattice for the KVBS.

More ML estimators will be added later: add entries to ESTIMATORS below.

Usage
-----
    conda activate phaselearner
    python paper/scripts/fig_phase_diagram.py             # -> paper/figures/fig_phase_diagram.pdf (+ .png)
    python paper/scripts/fig_phase_diagram.py --no-color  # plain (text region labels, no fills)
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

import paperstyle as ps

ROOT = Path(__file__).resolve().parents[2]
NPZ_DIR = ROOT / "results" / "npz"

# --- FSS reference beta_c (w=0.5, L=12; from sohan_phase_diagram.pdf) ---------
# Mirrors ssh_ml_phase/config.py FSS_DICT.  Treated as the phase boundary.
FSS_ALPHA = np.array([0.240, 0.245, 0.250, 0.255])
FSS_BETA_C = np.array([17.5, 14.3, 12.5, 12.3])
# Critical TEMPERATURE T_c = 1/beta_c (units of t). The guide-to-the-eye is the
# beta_c PCHIP INVERTED (1/beta_c(alpha)), NOT a fresh interpolation of the T_c
# points -- so this boundary is the SAME physical curve as the beta_c diagram
# (1/x is nonlinear, so the two interpolants differ between nodes; both still
# pass exactly through the 4 T_c nodes below). T_c rises 0.0571 -> 0.0813.
FSS_TC = 1.0 / FSS_BETA_C

# --- region colours (soft, print-safe; ordered=blue, disordered=sand) --------
KVBS_FILL = "#dbe7f5"        # light blue  -- ordered (KVBS) region
SEMIMETAL_FILL = "#f6ecd2"   # light sand  -- disordered (semimetal) region

# --- ML estimator series -----------------------------------------------------
# Each entry: how to pull a per-alpha beta_c out of a pca_results_*.npz file.
# beta_c array layout (ssh_ml_phase/pca_method.run_all):
#   [argmax_P1, argmax_P2, km_sig, km_sig_err, km_deriv, best, mean_P1_deriv_bc]
# The user-chosen estimator is km_sig (index 2), the k-means ordered-fraction
# sigmoid midpoint; its formal curve_fit error (index 3) is <0.01 (smaller than
# a marker) so it is not drawn as a visible error bar.
KM_SIG, KM_SIG_ERR = 2, 3

ESTIMATORS = [
    dict(
        label=r"PCA",
        # sID-5 sweep: has all five alpha incl. 0.2425 (the legacy
        # ..._position.npz only carries the original four).
        npz="pca_results_integrated_bond33_position_sid5.npz",
        bc_index=KM_SIG,
        err_index=KM_SIG_ERR,
        color=ps.OKABE_ITO[1],   # vermilion
        marker="s",
    ),
]


def load_estimator(npz_name, bc_index, err_index):
    """Return (alphas, beta_c, beta_c_err) sorted by alpha for one npz file."""
    d = np.load(NPZ_DIR / npz_name, allow_pickle=True)
    keys = sorted({k.split("_")[0] for k in d.files if k.startswith("a0.")})
    alphas, bc, err = [], [], []
    for key in keys:
        arr = d[f"{key}_beta_c"]
        alphas.append(float(key[1:]))
        bc.append(float(arr[bc_index]))
        err.append(float(arr[err_index]))
    order = np.argsort(alphas)
    return (np.array(alphas)[order], np.array(bc)[order], np.array(err)[order])


# ---------------------------------------------------------------------------
# Honeycomb cartoons: a uniform lattice cues the semimetal, a Kekule lattice
# (one sublattice of hexagons with strengthened bonds) cues the KVBS.
# ---------------------------------------------------------------------------
def _hex_vertices(center, R=1.0):
    """Flat-top hexagon: 6 vertices at 0,60,...,300 deg about ``center``."""
    ang = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack([center[0] + R * np.cos(ang),
                            center[1] + R * np.sin(ang)])


def _flower_centers(R=1.0):
    """Seven hexagon centres: one central + six neighbours (flat-top packing).

    Index 0 is the centre; indices 1..6 are the ring at 30,90,...,330 deg.
    """
    centers = [np.array([0.0, 0.0])]
    d = np.sqrt(3.0) * R
    for a in np.deg2rad(np.arange(30.0, 360.0, 60.0)):
        centers.append(np.array([d * np.cos(a), d * np.sin(a)]))
    return centers


def _honeycomb_bonds(R=1.0):
    """Unique bonds of the seven-hexagon flower.

    Returns (edges, verts) where ``edges`` maps a frozenset of two vertex keys
    to the set of hexagon indices that share that bond, and ``verts`` maps a
    vertex key to its (x, y) coordinates.
    """
    def key(p):
        return (round(float(p[0]), 3), round(float(p[1]), 3))

    edges, verts = {}, {}
    for hi, c in enumerate(_flower_centers(R)):
        vs = _hex_vertices(c, R)
        ks = [key(v) for v in vs]
        for v, k in zip(vs, ks):
            verts[k] = v
        for i in range(6):
            e = frozenset((ks[i], ks[(i + 1) % 6]))
            edges.setdefault(e, set()).add(hi)
    return edges, verts


def draw_honeycomb(host_ax, box, kekule, R=1.0):
    """Draw a small honeycomb cartoon inside ``box`` (axes-fraction bounds).

    kekule=False -> uniform lattice (semimetal); kekule=True -> alternating
    strong-bond Kekule lattice (KVBS), strong bonds on hexagons {1,3,5}.
    """
    iax = host_ax.inset_axes(box)
    iax.set_aspect("equal")
    iax.axis("off")
    iax.patch.set_visible(False)          # let the region colour show through

    edges, verts = _honeycomb_bonds(R)
    strong_hex = {1, 3, 5} if kekule else set()
    for e, hexset in edges.items():
        k1, k2 = tuple(e)
        (x1, y1), (x2, y2) = verts[k1], verts[k2]
        if kekule and (hexset & strong_hex):
            lw, col, z = 1.9, "#10243f", 3      # strong (dimerized) bond
        elif kekule:
            lw, col, z = 0.6, "#9bb0c4", 2      # weak bond
        else:
            lw, col, z = 1.0, "#3a3a3a", 2      # uniform bond (semimetal)
        iax.plot([x1, x2], [y1, y2], "-", color=col, lw=lw,
                 solid_capstyle="round", zorder=z)

    vv = np.array(list(verts.values()))
    iax.plot(vv[:, 0], vv[:, 1], "o", color="#1d1d1d", ms=1.7,
             markeredgecolor="none", zorder=4)
    m = 2.75
    iax.set_xlim(-m, m)
    iax.set_ylim(-m, m)
    return iax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-color", dest="color", action="store_false",
                   help="disable the region fills and phase cartoons "
                        "(plain text region labels only)")
    args = p.parse_args()

    ps.use()
    fig, ax = plt.subplots(figsize=ps.figsize())

    # Frame flush with the coloured phases: the x-window is set EXACTLY to the
    # shaded alpha range [0.240, 0.255] (user-requested), so the fills touch
    # both spines. The endpoint tick labels (0.240, 0.255) and the outer halves
    # of the endpoint marker clusters consequently sit on / overflow the frame
    # -- accepted on purpose. A little extra T headroom ABOVE the boundary gives
    # the semimetal region (now the upper region, in T) and its cartoon room to
    # breathe; the KVBS region is the wedge below the rising boundary.
    ax.set_xlim(0.240, 0.255)
    ax.set_ylim(0.048, 0.102)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()

    # Smooth, monotone boundary (guide to the eye). Interpolate beta_c with a
    # monotone PCHIP and INVERT it, so the T_c boundary is the same physical
    # curve as the beta_c diagram (see FSS_TC note); the FSS nodes at FSS_TC lie
    # exactly on it. beta_c falls monotonically, so 1/beta_c rises without
    # overshoot.
    boundary_beta = PchipInterpolator(FSS_ALPHA, FSS_BETA_C, extrapolate=False)

    def boundary(a):
        return 1.0 / boundary_beta(a)

    a_data = np.linspace(FSS_ALPHA.min(), FSS_ALPHA.max(), 400)

    # --- coloured phase regions, split by the boundary -----------------------
    # Shade ONLY across the determined alpha range [0.240, 0.255]; the thin
    # margins stay white so the diagram never implies a phase boundary where no
    # FSS point constrains it (no flat extrapolation past the end couplings).
    if args.color:
        b_data = boundary(a_data)
        # KVBS (ordered) is now BELOW the boundary (low T); semimetal ABOVE it.
        ax.fill_between(a_data, ylo, b_data, color=KVBS_FILL, lw=0, zorder=0)
        ax.fill_between(a_data, b_data, yhi, color=SEMIMETAL_FILL, lw=0, zorder=0)

    # --- ML estimator series: discrete markers (no connecting line) so the ----
    # comparison against the FSS boundary stays unambiguous as estimators grow.
    # Drawn BELOW the FSS reference node (see below) so that where an estimate
    # nearly coincides with FSS (the endpoint couplings) the small FSS dot nests
    # cleanly INSIDE the larger estimator marker -- a deliberate "ML agrees with
    # FSS" target -- instead of two equal outlines fusing into one glyph.
    for est in ESTIMATORS:
        a, bc, _err = load_estimator(est["npz"], est["bc_index"], est["err_index"])
        tc = 1.0 / bc                       # beta_c -> T_c (units of t)
        ax.plot(a, tc, linestyle="none", color=est["color"], zorder=5,
                marker=est["marker"], markersize=4.3, markerfacecolor="white",
                markeredgecolor=est["color"], label=est["label"])

    # --- FSS phase boundary: smooth black line + small reference nodes ---------
    # The nodes are a deliberately smaller white dot than the estimator markers
    # and drawn on top, so a coincident estimate reads as a bullseye (FSS node
    # within the estimator square) rather than an ambiguous "padlock".
    ax.plot(a_data, boundary(a_data), "-", color="black", lw=1.0, zorder=3)
    ax.plot(FSS_ALPHA, FSS_TC, linestyle="none", color="black", zorder=6,
            marker="o", markersize=2.8, markerfacecolor="white",
            markeredgecolor="black", label="FSS")

    # --- phase labels + cartoons ---------------------------------------------
    # In T (unlike beta) the ordered KVBS is the wedge BELOW the rising boundary
    # (widest at the lower right); the semimetal is the large region ABOVE it
    # (widest at the upper left).
    label_kw = dict(fontsize=8.5, color="0.15", zorder=6)
    if args.color:
        ax.text(0.80, 0.45, "KVBS", transform=ax.transAxes,
                ha="center", va="bottom", **label_kw)
        draw_honeycomb(ax, [0.68, 0.06, 0.24, 0.34], kekule=True)
        ax.text(0.185, 0.95, "semimetal", transform=ax.transAxes,
                ha="center", va="top", **label_kw)
        draw_honeycomb(ax, [0.07, 0.52, 0.23, 0.31], kekule=False)
    else:
        ax.text(0.96, 0.06, "KVBS", transform=ax.transAxes,
                ha="right", va="bottom", **label_kw)
        ax.text(0.04, 0.93, "semimetal", transform=ax.transAxes,
                ha="left", va="top", **label_kw)

    # --- in-panel run parameter (PRX convention, cf. fig_kvbs_*) --------------
    # Only omega is shown in-panel; L=12 was dropped at the user's request.
    ax.text(0.035, 0.045, r"$\omega=0.5$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$T_c$")
    ax.set_xticks([0.240, 0.245, 0.250, 0.255])
    ax.set_xticklabels(["0.240", "0.245", "0.250", "0.255"])
    ax.set_yticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    # Legend sits in the open upper-right (semimetal region, above the rising
    # boundary and clear of the region labels: semimetal upper-left, KVBS
    # lower-right). FSS is plotted last (top z-order, for the nesting) but listed
    # first in the legend -- ground truth before estimators -- so reorder here.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] != "FSS")
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper right", bbox_to_anchor=(0.99, 0.99), handlelength=1.4)

    ps.save(fig, "fig_phase_diagram")


if __name__ == "__main__":
    main()
