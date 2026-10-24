"""
Physics observables vs T (and the twin vs β) for the 2D attractive
Hubbard pilot. Reads SmoQyDQMC's per-β jackknife outputs only — no
manual binning or HDF5 traversal.

Sources per β (each `data/hubbard_square/sim_data/<β-folder>/`):
  - global_stats.csv  rows  `compressibility`, `double_occ`        → κ, D
  - integrated/density/density_momentum_integrated_stats.csv (q=0) → χ_den(q=0)
  - equal-time/density/density_momentum_equal-time_stats.csv (π,π) → S_den(π,π)
  - integrated/pair/pair_momentum_integrated_stats.csv      (q=0)  → χ_sc(q=0)
  - T·χ_den(q=0)                                                   → χ_den(q=0)/β

Outputs (data/hubbard_square/figures/):
  - observables_vs_T.pdf       — 2×3 panel grid vs T = 1/β
  - observables_vs_beta.pdf    — same observables vs β
  - derivatives_vs_T.pdf       — numerical d/dT of each observable
  - derivatives_vs_beta.pdf    — numerical d/dβ of each observable
"""
import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":           True,
    "font.family":           "serif",
    "font.serif":            ["Computer Modern Roman"],
    "axes.titlesize":        15,
    "axes.labelsize":        14,
    "axes.linewidth":        0.8,
    "axes.spines.top":       True,
    "axes.spines.right":     True,
    "axes.spines.left":      True,
    "axes.spines.bottom":    True,
    "axes.grid":             True,
    "grid.alpha":            0.25,
    "grid.linewidth":        0.5,
    "legend.fontsize":       11,
    "legend.frameon":        False,
    "xtick.labelsize":       12,
    "ytick.labelsize":       12,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      4,
    "ytick.major.size":      4,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "text.latex.preamble":   r"\usepackage{amsmath}\usepackage{amssymb}",
})

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT     = os.path.join(PROJECT_ROOT, "data", "hubbard_square", "sim_data")
OUT_DIR      = os.path.join(PROJECT_ROOT, "data", "hubbard_square", "figures")
NPZ_DIR      = os.path.join(PROJECT_ROOT, "results", "npz")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pilot constants ───────────────────────────────────────────────────────────
L         = 8
ABS_U     = 4.0
N_TARGET  = 0.875

# ── Reference T_c / β_c (Paiva 2004 PRG; digitized) ───────────────────────────
LIT_BC_CENTRAL = 7.00
LIT_BC_LO      = 6.54
LIT_BC_HI      = 7.51
LIT_TC_CENTRAL = 1.0 / LIT_BC_CENTRAL
LIT_TC_LO      = 1.0 / LIT_BC_HI    # T-band lower = 1/β-band upper
LIT_TC_HI      = 1.0 / LIT_BC_LO

PCA_SOURCE_OBS = "integrated_density_momentum"


def load_pca_betac():
    npz = os.path.join(NPZ_DIR, f"pca_results_{PCA_SOURCE_OBS}.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=False)
    cx = d.get("a4.0000_crossings_P1P2")
    if cx is None or cx.size == 0 or not np.isfinite(cx[0]):
        return None
    return float(cx[0])


# ── SmoQyDQMC stat CSV readers ────────────────────────────────────────────────

def read_global_stats(sim_dir):
    """Return {measurement: (MEAN_REAL, STD)} from global_stats.csv."""
    fp = os.path.join(sim_dir, "global_stats.csv")
    if not os.path.exists(fp):
        return {}
    out = {}
    with open(fp) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            out[parts[0]] = (float(parts[1]), float(parts[3]))
    return out


def read_correlator(sim_dir, channel, obs, space, K_indices):
    """Return (MEAN_REAL, STD) at the (K_1, K_2) component of the requested
    momentum-space correlator stats CSV.  channel ∈ {'integrated','equal-time'};
    obs ∈ {'density','pair','spin_z',...}; space ∈ {'momentum','position'}."""
    fname = f"{obs}_{space}_{channel}_stats.csv"
    fp    = os.path.join(sim_dir, channel, obs, fname)
    if not os.path.exists(fp):
        return None
    K1, K2 = K_indices
    # Column layout: INDEX  ORBITAL_ID_2  ORBITAL_ID_1  K_2  K_1  MEAN_REAL  MEAN_IMAG  STD
    with open(fp) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                k_2 = int(parts[3]); k_1 = int(parts[4])
            except ValueError:
                continue
            if k_1 == K1 and k_2 == K2:
                return float(parts[5]), float(parts[7])
    return None


# ── Per-β assembly ────────────────────────────────────────────────────────────
_BETA_RE = re.compile(r"_b([0-9]+\.[0-9]+)-1$")


def list_beta_dirs():
    out = []
    for d in sorted(glob.glob(os.path.join(SIM_ROOT, "hubbard_square_U-*"))):
        m = _BETA_RE.search(d)
        if m:
            out.append((float(m.group(1)), d))
    return sorted(out)


def compute_beta_point(sim_dir, beta):
    chi_den = read_correlator(sim_dir, "integrated", "density", "momentum", (0, 0))
    S_pipi  = read_correlator(sim_dir, "equal-time",  "density", "momentum",
                              (L // 2, L // 2))
    chi_sc  = read_correlator(sim_dir, "integrated", "pair",    "momentum", (0, 0))
    if any(x is None for x in (chi_den, S_pipi, chi_sc)):
        return None
    gs = read_global_stats(sim_dir)
    if "compressibility" not in gs or "double_occ" not in gs:
        return None
    return {
        "beta":       beta,
        "T":          1.0 / beta,
        "chi_den":    chi_den,
        "Tchi_den":   (chi_den[0] / beta, chi_den[1] / beta),
        "kappa":      gs["compressibility"],
        "S_pipi":     S_pipi,
        "chi_sc":     chi_sc,
        "double_occ": gs["double_occ"],
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────
DATA_COLOR  = "#1f4e9c"   # deep blue
DATA_EDGE   = "white"
REF_COLOR   = "0.45"      # mid-gray
PCA_COLOR   = "#c0392b"   # crimson


def _draw_refs(ax, ref_central, ref_lo, ref_hi, pca_x, *,
               show_pca=True, show_lit=True, show_band=True, label=False):
    if show_band:
        ax.axvspan(ref_lo, ref_hi, color=REF_COLOR, alpha=0.13, ec="none",
                   label=("lit.\\ band" if label else None))
    if show_lit:
        ax.axvline(ref_central, color=REF_COLOR, ls=":", lw=1.1,
                   label=(rf"lit.\\ ${ref_central:.3f}$" if label else None))
    if show_pca and pca_x is not None:
        ax.axvline(pca_x, color=PCA_COLOR, ls="--", lw=1.4,
                   label=(rf"PCA ${pca_x:.3f}$" if label else None))


def _draw_data(ax, x, m, se, *, marker="o", lw_line=1.2, alpha_line=0.85):
    ax.plot(x, m, "-", lw=lw_line, color=DATA_COLOR, alpha=alpha_line,
            zorder=2)
    ax.errorbar(x, m, yerr=se, fmt=marker, ms=4.6,
                color=DATA_COLOR, ecolor=DATA_COLOR,
                markeredgecolor=DATA_EDGE, markeredgewidth=0.6,
                elinewidth=0.9, capsize=2, capthick=0.8, zorder=3)


def _make_panel(ax, x, m, se, title, xlab, *,
                ref_central, ref_lo, ref_hi, pca_x,
                show_legend=False):
    _draw_refs(ax, ref_central, ref_lo, ref_hi, pca_x, label=show_legend)
    _draw_data(ax, x, m, se)
    ax.set_xlabel(xlab)
    ax.set_title(title, pad=6)
    if show_legend:
        ax.legend(loc="best", ncol=1)


# ── Figure assembly ───────────────────────────────────────────────────────────

PANEL_SPECS = [
    ("chi_den",    r"$\chi_{\rm den}(q\!=\!0)$"),
    ("Tchi_den",   r"$T\,\chi_{\rm den}(q\!=\!0)$"),
    ("kappa",      r"$\kappa = \frac{1}{N}\,\partial n/\partial\mu$"),
    ("S_pipi",     r"$S_{\rm den}(\pi,\pi)$"),
    ("chi_sc",     r"$\chi_{\rm sc}(q\!=\!0)$"),
    ("double_occ", r"$D = \langle n_{i\uparrow} n_{i\downarrow}\rangle$"),
]


def _column(rows, key):
    return (np.array([r[key][0] for r in rows]),
            np.array([r[key][1] for r in rows]))


def numerical_derivative(x, y, se):
    """Central-difference derivative dy/dx + stderr propagation, assuming
    y_i are independent (each is a separate-β jackknife mean).  Edge points
    fall back to one-sided forward/backward differences."""
    n = len(x)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    dy  = np.gradient(y, x)
    dse = np.empty(n)
    dse[0]  = np.hypot(se[0],  se[1])  / abs(x[1]  - x[0])
    dse[-1] = np.hypot(se[-1], se[-2]) / abs(x[-1] - x[-2])
    for i in range(1, n - 1):
        dse[i] = np.hypot(se[i + 1], se[i - 1]) / abs(x[i + 1] - x[i - 1])
    return dy, dse


def make_figure(rows, x_key, x_lab, ref_central, ref_lo, ref_hi, pca_x,
                title_axis, out_path):
    rows = sorted(rows, key=lambda r: r[x_key])
    x = np.array([r[x_key] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.0),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.18,
                                    wspace=0.10, hspace=0.10)
    for idx, (ax, (key, title)) in enumerate(zip(axes.flat, PANEL_SPECS)):
        m, se = _column(rows, key)
        _make_panel(ax, x, m, se, title, x_lab,
                    ref_central=ref_central, ref_lo=ref_lo,
                    ref_hi=ref_hi, pca_x=pca_x,
                    show_legend=(idx == 0))
        ax.set_box_aspect(1)   # force the data-area box to be square

    # Pin the T·χ panel's y-axis to [45, 55] so the ~49 trace sits cleanly
    # in the middle without the auto-scale zooming in to where every bin
    # of stat noise looks like a feature.
    axes.flat[1].set_ylim(45, 55)

    fig.suptitle(rf"Attractive Hubbard --- "
                 rf"$U\!=\!-{ABS_U:.0f}$, $\langle n\rangle\!=\!{N_TARGET}$, "
                 rf"$L\!=\!{L}$  $|$  vs.\ {title_axis}",
                 fontsize=17, y=1.01)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_derivative_figure(rows, x_key, x_lab, ref_central, ref_lo, ref_hi,
                           pca_x, title_axis, out_path):
    """Numerical d(obs)/dx (where x ∈ {T, β}) for each of the six panels."""
    rows = sorted(rows, key=lambda r: r[x_key])
    x = np.array([r[x_key] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.0),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.18,
                                    wspace=0.10, hspace=0.10)

    d_var = "T" if x_key == "T" else r"\beta"
    for idx, (ax, (key, title)) in enumerate(zip(axes.flat, PANEL_SPECS)):
        m, se   = _column(rows, key)
        dm, dse = numerical_derivative(x, m, se)
        d_title = rf"$d/d{d_var}\;$ {title}"
        _make_panel(ax, x, dm, dse, d_title, x_lab,
                    ref_central=ref_central, ref_lo=ref_lo,
                    ref_hi=ref_hi, pca_x=pca_x,
                    show_legend=(idx == 0))
        ax.axhline(0, color="black", lw=0.7, alpha=0.5, zorder=1)
        ax.set_box_aspect(1)

    fig.suptitle(rf"Attractive Hubbard --- $U\!=\!-{ABS_U:.0f}$, "
                 rf"$\langle n\rangle\!=\!{N_TARGET}$, $L\!=\!{L}$  $|$  "
                 rf"derivatives vs.\ {title_axis}",
                 fontsize=17, y=1.01)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pca_betac = load_pca_betac()
    if pca_betac is None:
        print(f"  WARN: PCA β_c unavailable from {PCA_SOURCE_OBS} npz")
        pca_tc = None
    else:
        pca_tc = 1.0 / pca_betac
        print(f"PCA β_c (P1↔P2 crossing on {PCA_SOURCE_OBS}) = {pca_betac:.3f}  "
              f"→  T_c = {pca_tc:.4f}")
    print(f"Lit β_c (Paiva 04 PRG): {LIT_BC_CENTRAL:.2f}  "
          f"band [{LIT_BC_LO:.2f}, {LIT_BC_HI:.2f}]  "
          f"→  T_c = {LIT_TC_CENTRAL:.4f}  band [{LIT_TC_LO:.4f}, {LIT_TC_HI:.4f}]")

    rows = []
    for beta, d in list_beta_dirs():
        row = compute_beta_point(d, beta)
        if row is None:
            print(f"  β={beta:.2f}: missing CSV/stats, skipping")
            continue
        rows.append(row)

    if not rows:
        print("No valid β points found; aborting.")
        return

    print(f"\nLoaded {len(rows)} β points.\n")

    # Plot vs T
    make_figure(
        rows, x_key="T", x_lab=r"$T$",
        ref_central=LIT_TC_CENTRAL, ref_lo=LIT_TC_LO, ref_hi=LIT_TC_HI,
        pca_x=pca_tc, title_axis=r"$T = 1/\beta$",
        out_path=os.path.join(OUT_DIR, "observables_vs_T.pdf"),
    )

    # Plot vs β
    make_figure(
        rows, x_key="beta", x_lab=r"$\beta$",
        ref_central=LIT_BC_CENTRAL, ref_lo=LIT_BC_LO, ref_hi=LIT_BC_HI,
        pca_x=pca_betac, title_axis=r"$\beta$",
        out_path=os.path.join(OUT_DIR, "observables_vs_beta.pdf"),
    )

    # Derivatives vs T
    make_derivative_figure(
        rows, x_key="T", x_lab=r"$T$",
        ref_central=LIT_TC_CENTRAL, ref_lo=LIT_TC_LO, ref_hi=LIT_TC_HI,
        pca_x=pca_tc, title_axis=r"$T = 1/\beta$",
        out_path=os.path.join(OUT_DIR, "derivatives_vs_T.pdf"),
    )

    # Derivatives vs β
    make_derivative_figure(
        rows, x_key="beta", x_lab=r"$\beta$",
        ref_central=LIT_BC_CENTRAL, ref_lo=LIT_BC_LO, ref_hi=LIT_BC_HI,
        pca_x=pca_betac, title_axis=r"$\beta$",
        out_path=os.path.join(OUT_DIR, "derivatives_vs_beta.pdf"),
    )


if __name__ == "__main__":
    main()
