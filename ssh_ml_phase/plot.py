"""
Plotting pipeline for ML phase detection results.

Reads .npz files produced by pca_method / umap_method and generates
multi-page PDFs, one page per alpha (beta-sweep) or per beta (alpha-sweep).

Beta-sweep PDFs (under results/figures/<model>/<method>/<tag>/):
  pca/  avg_analysis.pdf  — 4×1 column: P1/P2 vs β, ⟨Pk⟩/σk overlay,
                             Var(Pk|β)/λk overlay (susceptibility)
  pca/  embedding.pdf     — 2×2: (P1,P2) scatter by β / by cluster,
                             f_ord sigmoid, df_ord/dβ
  pca/  PC_layout.pdf     — 3 stacked: PC1 loading, P1 ridgeline KDE,
                             eigenvalue spectrum
  umap/ embedding.pdf     — 2×2 mirror of pca/embedding.pdf, but on
                             UMAP coordinates instead of (P1, P2)
  summary/ beta_c_summary.pdf — β_c vs α across methods (FSS reference)

Alpha-sweep PDFs:
  pca_alpha_sweep_<tag>.pdf     — Var(α) + f_CDW + derivative
  umap_alpha_sweep_<tag>.pdf    — UMAP embedding + f_CDW + derivative

Usage (standalone):
  python3.11 -m ssh_ml_phase.plot --model bond_holstein --alphas 1.4142
  python3.11 -m ssh_ml_phase.plot --model bond_holstein --alpha-sweep --betas 0.35
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman"],
    "axes.titlesize":  11,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

from . import config as _cfg
from .data import make_tag, result_filename


# Legacy susceptibility (3-panel Var) and KMeans (4-panel scatter) PDFs.
# Kept around behind this flag so we don't lose the km_sig / sm_argmax_P1
# estimator overlay during the diagnostic-style rollout.  Set True to also
# emit pca_susceptibility.pdf and pca_kmeans.pdf alongside the new
# avg_analysis.pdf / embedding.pdf / PC_layout.pdf trio.
LEGACY_VAR_KM_PDFS = False


# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  —  edit here to plot without using the command line
# ══════════════════════════════════════════════════════════════════════════════

MODEL       = "bond_holstein"              # preset: "ssh" | "holstein" | "bond_holstein"
ALPHAS      = [1.4142]                     # None → use all preset alphas
TAG         = "electron_densities_w1.00"   # must match the suffix of the .npz filename
BETA_MIN    = None                         # used only for the info label on plots
BETA_MAX    = None
ALPHA_SWEEP = False                        # True → plot alpha-sweep results
BETAS       = None                         # fixed β value(s) for alpha-sweep, e.g. [0.35]
SKIP_UMAP   = False                        # True → skip UMAP plots
RESULTS_DIR_OVERRIDE = None                # None → use default results/ directory

# ══════════════════════════════════════════════════════════════════════════════


# ── Style constants (match notebook conventions) ──────────────────────────────
_CM_BETA    = "viridis"    # continuous parameter (β or α) colormap —
                           # sequential (no white midpoint) so every point
                           # stays visible against the white background
_CM_CLUSTER = "Set1"       # KMeans cluster labels
_C_VAR1     = "#e41a1c"    # Var(P1) / primary estimate — red
_C_VAR2     = "#377eb8"    # Var(P2) / secondary        — blue
_C_FSS      = "#1a1aff"    # FSS reference              — blue dashed
_C_SIG      = "#e41a1c"    # sigmoid estimate           — red dashed
_C_DERIV    = "#ff7f00"    # derivative-peak estimate   — orange dotted
_C_SM       = "#4dac26"    # smoothed-argmax estimate   — dark-green dash-dot

# Diagnostic-panel palette (matches plot_pca_diagnostics.py)
_D_P1       = "#d62728"    # P1 / Var(P1)               — red
_D_P2       = "#2ca02c"    # P2 / Var(P2)               — green
_D_SCATTER  = "#1f77b4"    # per-sample scatter / spectrum — blue


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _sigmoid(x, xc, w):
    return 1.0 / (1.0 + np.exp(-(x - xc) / w))


def _figures_dir(tag, method, model, results_dir=None):
    """Return ``<results_dir>/figures/<model>/<method>/<tag>/``.

    ``method`` is the analysis stage — ``"pca"``, ``"umap"``, ``"summary"``,
    or any short label used by diagnostic scripts (``"diagnostics"``,
    ``"clustering_comparison"``, ``"correlations"``, …).  ``model`` falls
    back to ``"unspecified"``; ``tag`` is appended when non-empty so a
    bare model+method root holds aggregate plots.

    Tags ending in ``_nX.XXX`` (Hubbard-style filling suffix) are split so
    the filling becomes a parent directory:
        ``pair_momentum_n0.500`` → ``n0.500/pair_momentum``
    This keeps a single ⟨n⟩ value's plots together.
    """
    base = results_dir or _cfg.RESULTS_DIR
    parts = [base, "figures", model or "unspecified", method]
    if tag:
        m = re.search(r"_(n\d+\.\d+)$", tag)
        if m:
            parts.extend([m.group(1), tag[:m.start()]])
        else:
            parts.append(tag)
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def _info_str(mi):
    """Compact one-line parameter string for suptitles (LaTeX-safe)."""
    obs = mi.get("observable")
    if obs is not None:
        obs = str(obs).replace("_", r"\_")
    pairs = [("model",          mi.get("model")),
             ("obs",            obs),
             ("L",              mi.get("L")),
             (r"$\beta_{\min}$", mi.get("beta_min")),
             (r"$\beta_{\max}$", mi.get("beta_max"))]
    return "  $|$  ".join(f"{k}={v}" for k, v in pairs if v is not None)


def _ordered_label(model):
    """LaTeX label for the ordered-phase fraction, by model.

    Used both as y-axis label and inside titles for the 'cluster fraction'
    panels of PCA / UMAP. Returns a $...$ math string ready for matplotlib.
    """
    m = (model or "").lower()
    if m in ("attractive_hubbard", "hubbard", "negu"):
        return r"$f_\mathrm{SC}$"
    if m == "holstein":
        return r"$f_\mathrm{CDW}$"
    if m in ("ssh", "bond_holstein", ""):
        return r"$f_\mathrm{KVBS}$"
    return r"$f_\mathrm{ord}$"


def _vl(ax, x, color, ls, lw, label):
    """Draw one vertical line if x is a finite number."""
    if x is None:
        return
    try:
        if np.isnan(float(x)):
            return
    except (TypeError, ValueError):
        return
    ax.axvline(float(x), color=color, ls=ls, lw=lw, label=label, alpha=0.85)


def _vlines(ax, bc_arr=None, fss=None,
            idx_sm=5, idx_sig=2, idx_deriv=4,
            param_label="β"):
    """Overlay all standard estimator vertical lines on an axis."""
    if bc_arr is not None:
        if idx_sm is not None:
            _vl(ax, float(bc_arr[idx_sm]),
                _C_SM,    "-.", 1.3, f"sm argmax  {float(bc_arr[idx_sm]):.3f}")
        if idx_sig is not None:
            _vl(ax, float(bc_arr[idx_sig]),
                _C_SIG,   "--", 1.5, f"KM sig  {float(bc_arr[idx_sig]):.3f}")
        if idx_deriv is not None:
            _vl(ax, float(bc_arr[idx_deriv]),
                _C_DERIV, ":",  1.3, f"KM deriv  {float(bc_arr[idx_deriv]):.3f}")
    if fss is not None:
        _vl(ax, fss, _C_FSS, "--", 1.8, f"FSS  {fss:.3f}")


def _safe(data, key, default=None):
    return data[key] if key in data else default


# ── Shared panel builders ──────────────────────────────────────────────────────

def _panel_var(ax, u_x, var1, var1_sm, var2, var2_sm,
               bc_arr=None, fss=None, xlabel=r"$\beta$",
               idx_sm=5, idx_sig=2, idx_deriv=4):
    """Var(P1) and Var(P2) vs control parameter."""
    ax.plot(u_x, var1,    "o", ms=4, color=_C_VAR1, alpha=0.5)
    ax.plot(u_x, var1_sm, "-", lw=2, color=_C_VAR1, label=r"$\mathrm{Var}(P_1)$")
    ax.plot(u_x, var2,    "s", ms=4, color=_C_VAR2, alpha=0.5)
    ax.plot(u_x, var2_sm, "-", lw=2, color=_C_VAR2, label=r"$\mathrm{Var}(P_2)$")
    _vlines(ax, bc_arr, fss, idx_sm=idx_sm, idx_sig=idx_sig,
            idx_deriv=idx_deriv, param_label=xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{Var}(P)$")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)


def _panel_deriv(ax, d_x, d_var1, d_var2, xlabel=r"$\beta$"):
    """Derivative of smoothed Var(P1) and Var(P2)."""
    xl = xlabel.strip("$").strip("\\")   # bare symbol for subscript labels
    ax.plot(d_x, d_var1, "-", lw=2, color=_C_VAR1,
            label=r"$d\,\mathrm{Var}(P_1)/d" + xl + r"$")
    ax.plot(d_x, d_var2, "-", lw=2, color=_C_VAR2,
            label=r"$d\,\mathrm{Var}(P_2)/d" + xl + r"$")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ylim = max(np.abs(d_var1).max(), np.abs(d_var2).max()) * 1.2
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$d\,\mathrm{Var}/d" + xl + r"$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)


def _panel_scatter(ax, P1, P2, c_vals, cmap, c_label, title="",
                   xlabel="dim 1", ylabel="dim 2"):
    """Scatter P1 vs P2 (or emb_x vs emb_y) coloured by c_vals."""
    sc = ax.scatter(P1, P2, c=c_vals, cmap=cmap, s=1, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label=c_label, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _panel_f_ordered(ax, u_x, f_ord, popt, bc_arr, fss,
                     xlabel=r"$\beta$", ylabel=r"$f_\mathrm{KVBS}$",
                     idx_sm=None, idx_sig=2, idx_deriv=4):
    """f_ordered vs control parameter with sigmoid fit."""
    ax.plot(u_x, f_ord, "ko", ms=5)
    if popt is not None:
        x_fine = np.linspace(u_x[0], u_x[-1], 300)
        ax.plot(x_fine, _sigmoid(x_fine, popt[0], popt[1]),
                "r-", lw=2,
                label=rf"sigmoid fit  $\beta_c={popt[0]:.2f}\pm{popt[1]:.2f}$")
    _vlines(ax, bc_arr, fss,
            idx_sm=idx_sm, idx_sig=idx_sig, idx_deriv=idx_deriv,
            param_label=xlabel)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)


def _panel_f_deriv(ax, d_x, d_vals, popt, xlabel=r"$\beta$"):
    """Point-wise derivative of f_ordered, with analytic sigmoid derivative overlaid."""
    xl = xlabel.strip("$").strip("\\")
    ax.plot(d_x, d_vals, "ko-", ms=4, lw=1,
            label=r"$df/d" + xl + r"$ (point-wise)")
    if popt is not None:
        x_fine = np.linspace(d_x[0], d_x[-1], 300)
        s = _sigmoid(x_fine, popt[0], popt[1])
        ax.plot(x_fine, s * (1 - s) / popt[1], "r-", lw=2,
                label="analytic (sigmoid)")
    peak = d_x[np.argmax(d_vals)]
    _vl(ax, peak, _C_DERIV, ":", 1.4, rf"peak  {peak:.3f}")
    ax.set_ylim(-0.5, max(0.5, d_vals.max() * 1.4))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$df/d" + xl + r"$")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)


# ── Diagnostic-style panel builders ───────────────────────────────────────────
# Ported from plot_pca_diagnostics.py; β_c reference is the FSS value (not
# hard-coded) and skipped when NaN.

# Module-level: optional uncertainty band per FSS β_c value.  Populated
# at the start of plot_{pca,umap}_beta_sweep loops; consulted by
# _betac_vline so we don't have to plumb the band through every panel.
_BETAC_BANDS: dict = {}


def _betac_vline(ax, betac, *, lw=1.0, label=True):
    """Vertical dotted line at FSS β_c (+ shaded band if registered in
    `_BETAC_BANDS`) plus an optional small label.  No-op if betac is
    NaN or None."""
    if betac is None or not np.isfinite(betac):
        return
    band = _BETAC_BANDS.get(float(betac))
    if band is not None:
        lo, hi = band
        ax.axvspan(lo, hi, color="gray", alpha=0.15, ec="none")
    ax.axvline(betac, color="gray", ls=":", lw=lw)
    if label:
        ax.text(betac, 0.98, rf"\ $\beta_c\!\approx\!{betac:.2f}$",
                color="gray", fontsize=8, va="top", ha="left",
                transform=ax.get_xaxis_transform())


def _diag_panel_pc_vs_beta(ax, betas, P_samples, u_beta, betac, ylabel, title):
    """Per-sample scatter of (β, PC) plus per-β mean ± std band.

    The inset (per-β mean alone) and the legend are placed in the two
    lowest-density quadrants of the auto-scaled axes — y-limits are not
    modified, so wide std bands at intermediate β still set the y-extent.
    """
    ax.scatter(betas, P_samples, s=2, alpha=0.15, color=_D_SCATTER,
               rasterized=True, label=r"per-sample")
    means = np.array([P_samples[betas == b].mean() for b in u_beta])
    stds  = np.array([P_samples[betas == b].std()  for b in u_beta])
    ax.plot(u_beta, means, "o-", color=_D_P1, lw=1.5, ms=4,
            label=r"$\langle\cdot\rangle_\beta\ \pm\ \mathrm{std}$")
    ax.fill_between(u_beta, means - stds, means + stds,
                    alpha=0.25, color=_D_P1)
    _betac_vline(ax, betac)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)

    # ── Pick inset + legend corners from the existing axis limits ────────
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xr, yr   = xhi - xlo, yhi - ylo

    inset_w, inset_h = 0.40, 0.36
    pad = 0.05
    corners = {
        "upper left":  (pad,                 1 - inset_h - pad),
        "upper right": (1 - inset_w - pad,   1 - inset_h - pad),
        "lower left":  (pad,                 pad),
        "lower right": (1 - inset_w - pad,   pad),
    }

    def _density(x_frac, y_frac, w, h):
        x_lo = xlo + x_frac * xr
        x_hi = xlo + (x_frac + w) * xr
        y_lo = ylo + y_frac * yr
        y_hi = ylo + (y_frac + h) * yr
        mask = ((betas >= x_lo) & (betas <= x_hi) &
                (P_samples >= y_lo) & (P_samples <= y_hi))
        return int(mask.sum())

    densities    = {n: _density(x, y, inset_w, inset_h)
                    for n, (x, y) in corners.items()}
    inset_corner = min(densities, key=densities.get)
    ix, iy       = corners[inset_corner]

    leg_w, leg_h = 0.32, 0.18
    leg_dens = {n: _density(x, y, leg_w, leg_h)
                for n, (x, y) in corners.items() if n != inset_corner}
    ax.legend(loc=min(leg_dens, key=leg_dens.get), fontsize=8)

    inset = ax.inset_axes([ix, iy, inset_w, inset_h])
    inset.plot(u_beta, means, "o-", color=_D_P1, lw=1.4, ms=3)
    _betac_vline(inset, betac, lw=0.8, label=False)
    inset.set_title(r"$\langle\cdot\rangle_\beta$ only",
                    fontsize=7, pad=2)
    inset.tick_params(labelsize=6, length=2)
    inset.grid(alpha=0.3)
    inset.set_xlabel(r"$\beta$", fontsize=7, labelpad=1)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.92)


def _diag_panel_pc_overlay(ax, P1_samples, P2_samples, betas, u_beta,
                           betac, title):
    """Overlay z-scored per-β means of P1, P2 and mark sign-change crossings.

    Each PC is divided by its own global std (≈ √λ_k) so the two share a
    fair scale.  Yellow markers sit on the curves at sign changes; the β_×
    values are placed in the upper-left corner.
    """
    s1 = float(P1_samples.std())
    s2 = float(P2_samples.std())
    m1 = np.array([P1_samples[betas == b].mean() for b in u_beta]) / s1
    m2 = np.array([P2_samples[betas == b].mean() for b in u_beta]) / s2

    ax.plot(u_beta, m1, "o-", color=_D_P1, lw=1.5, ms=4,
            label=r"$\langle P_1\rangle_\beta / \sigma_1$")
    ax.plot(u_beta, m2, "s-", color=_D_P2, lw=1.5, ms=4,
            label=r"$\langle P_2\rangle_\beta / \sigma_2$")

    diff = m1 - m2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    crossings = []
    for i in sign_changes:
        b0, b1 = u_beta[i], u_beta[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        if d1 == d0:
            continue
        b_cross = b0 - d0 * (b1 - b0) / (d1 - d0)
        y_cross = m1[i] + (m1[i + 1] - m1[i]) * (b_cross - b0) / (b1 - b0)
        ax.plot(b_cross, y_cross, "o", ms=9,
                mfc="yellow", mec="black", zorder=5)
        crossings.append(b_cross)

    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    _betac_vline(ax, betac)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle P_k\rangle_\beta / \sigma_k$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    if crossings:
        info = "\n".join(rf"$\beta_\times={b:.2f}$" for b in crossings)
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9))


def _diag_panel_pc_var_overlay(ax, var_P1, var_P2, P1_samples, P2_samples,
                               u_beta, betac, title):
    """Overlay Var(P_k|β)/λ_k for k=1,2; mark each argmax in yellow."""
    lambda1 = float(P1_samples.var())
    lambda2 = float(P2_samples.var())
    v1 = var_P1 / lambda1
    v2 = var_P2 / lambda2

    ax.plot(u_beta, v1, "o-", color=_D_P1, lw=1.5, ms=4,
            label=r"$\mathrm{Var}(P_1|\beta)/\lambda_1$")
    ax.plot(u_beta, v2, "s-", color=_D_P2, lw=1.5, ms=4,
            label=r"$\mathrm{Var}(P_2|\beta)/\lambda_2$")
    ax.axhline(1.0, color="black", lw=0.7, alpha=0.5, ls="--",
               label=r"global $= 1$")
    _betac_vline(ax, betac)

    bc1 = u_beta[np.argmax(v1)]
    bc2 = u_beta[np.argmax(v2)]
    ax.plot(bc1, v1.max(), "o", ms=10, mfc="yellow", mec=_D_P1, zorder=5)
    ax.plot(bc2, v2.max(), "s", ms=10, mfc="yellow", mec=_D_P2, zorder=5)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\mathrm{Var}(P_k|\beta)/\lambda_k$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax.text(0.02, 0.98, rf"$\beta_c^{{(1)}}={bc1:.1f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=_D_P1,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=_D_P1, alpha=0.9))
    ax.text(0.02, 0.88, rf"$\beta_c^{{(2)}}={bc2:.1f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=_D_P2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=_D_P2, alpha=0.9))


def _diag_panel_p1p2_scatter(ax, P1, P2, betas, betac,
                             xlabel=r"$P_1$", ylabel=r"$P_2$",
                             title=r"$(P_1, P_2)$ coloured by $\beta$"):
    """2D embedding scatter coloured by β with a sequential colormap centred
    at β_c (β_c maps to the middle of the colormap via TwoSlopeNorm).  Uses
    viridis — no white/near-white region — so points near β_c stay visible
    against the white background.  Falls back to a plain linear norm when β_c
    is unavailable.
    Generic over PCA scores and UMAP coordinates via the xlabel/ylabel hooks.
    """
    cmap = "viridis"
    if betac is not None and np.isfinite(betac):
        norm = TwoSlopeNorm(vcenter=float(betac),
                            vmin=float(betas.min()), vmax=float(betas.max()))
    else:
        norm = None
    sc = ax.scatter(P1, P2, c=betas, cmap=cmap, norm=norm,
                    s=5, alpha=0.7, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.set_title(r"$\beta$", fontsize=11, pad=6)
    if betac is not None and np.isfinite(betac):
        cb.ax.axhline(float(betac), color="black", lw=0.9)
        cb.ax.text(1.15, float(betac),
                   rf"$\beta_c={float(betac):.2f}$",
                   color="black", fontsize=8, ha="left", va="center",
                   transform=cb.ax.get_yaxis_transform())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


def _diag_panel_p1p2_clusters(ax, P1, P2, km_labels, betas,
                              xlabel=r"$P_1$", ylabel=r"$P_2$",
                              title=r"$(P_1, P_2)$ coloured by cluster"):
    """(P1,P2) scatter coloured by KMeans cluster — discrete legend (no
    colorbar).  The cluster with the higher mean-β is rendered blue
    (``ordered``); the other red (``disordered``).
    """
    km = km_labels.astype(int)
    if (km == 0).any() and (km == 1).any():
        mb0 = float(betas[km == 0].mean())
        mb1 = float(betas[km == 1].mean())
        ordered_label = 1 if mb1 > mb0 else 0
    else:
        ordered_label = 1
    mask_ord = km == ordered_label

    ax.scatter(P1[~mask_ord], P2[~mask_ord], c=_D_P1,      s=5, alpha=0.6,
               rasterized=True, label=r"disordered")
    ax.scatter(P1[mask_ord],  P2[mask_ord],  c=_D_SCATTER, s=5, alpha=0.6,
               rasterized=True, label=r"ordered")
    leg = ax.legend(loc="best", fontsize=9, framealpha=0.9, markerscale=2.5,
                    handletextpad=0.4, borderpad=0.4)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


def _diag_panel_f_sigmoid(ax, u_beta, f_ord, popt, betac, ylabel,
                          title=None, xlabel=r"$\beta$"):
    """Order-parameter cluster fraction f_ord(β) with sigmoid fit, in the
    diagnostic palette (black markers, red fit, gray FSS reference).  The
    sigmoid β_c±err is shown in a corner box."""
    ax.plot(u_beta, f_ord, "o", color="black", ms=5, label=r"data")
    if popt is not None:
        x_fine = np.linspace(u_beta[0], u_beta[-1], 300)
        ax.plot(x_fine, _sigmoid(x_fine, popt[0], popt[1]),
                "-", lw=2, color=_D_P1, label=r"sigmoid fit")
        ax.axvline(float(popt[0]), color=_D_P2, ls="--", lw=1.4,
                   alpha=0.85, label=r"$\beta_c^{\mathrm{sig}}$")
    _betac_vline(ax, betac)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if popt is not None:
        ax.text(0.98, 0.04,
                rf"$\beta_c^{{\mathrm{{sig}}}}={popt[0]:.2f}\pm{popt[1]:.2f}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=_D_P1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=_D_P1, alpha=0.9))
    ax.set_title(title or rf"{ylabel} vs $\beta$  (KMeans sigmoid)",
                 fontsize=11)


def _diag_panel_f_deriv(ax, d_beta, d_vals, popt, betac, ylabel,
                        title=None, xlabel=r"$\beta$"):
    """Point-wise derivative of f_ord with the analytic sigmoid derivative
    overlaid; argmax in yellow.  Same diagnostic styling as
    `_diag_panel_pc_var_overlay` so the bottom row matches the avg_analysis
    visual vocabulary."""
    ax.plot(d_beta, d_vals, "o", color="black", ms=4, label=r"point-wise")
    if popt is not None:
        x_fine = np.linspace(d_beta[0], d_beta[-1], 300)
        s = _sigmoid(x_fine, popt[0], popt[1])
        ax.plot(x_fine, s * (1 - s) / popt[1], "-", lw=2, color=_D_P1,
                label=r"analytic (sigmoid)")
    peak_idx = int(np.argmax(d_vals))
    peak_b   = float(d_beta[peak_idx])
    peak_v   = float(d_vals[peak_idx])
    ax.plot(peak_b, peak_v, "o", ms=11,
            mfc="yellow", mec="black", zorder=5)
    ax.axvline(peak_b, color=_D_P2, ls="--", lw=1.4, alpha=0.85,
               label=r"$\beta_c^{\mathrm{deriv}}$")
    _betac_vline(ax, betac)
    ax.set_xlabel(xlabel)
    ord_inner = ylabel.strip("$")
    xl = xlabel.strip("$").strip("\\")
    ax.set_ylabel(rf"$d{ord_inner}/d{xl}$")
    ymax = float(np.max(d_vals)) if len(d_vals) else 1.0
    ax.set_ylim(-0.5, max(0.5, ymax * 1.4))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.98, 0.96, rf"peak $\beta={peak_b:.2f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.9))
    ax.set_title(title or rf"$d{ord_inner}/d{xl}$  "
                 r"(transition sharpness)",
                 fontsize=11)


def _diag_panel_pc_loading(ax, loading, L, is_q, label, pc_index=1):
    """PC{k} loading — L×L imshow on the lattice or BZ when len(loading)=L²
    (or L²−1, i.e. the origin/self-term column was excluded by a transform);
    otherwise a feature-index bar plot of |loading|."""
    if loading is None or len(loading) == 0:
        ax.set_visible(False)
        return
    pc_lab = f"PC{pc_index}"
    grid = None
    if len(loading) == L * L:
        grid = np.asarray(loading, float)
    elif len(loading) == L * L - 1:            # origin (index 0) was excluded
        grid = np.full(L * L, np.nan)
        grid[1:] = loading
    if grid is not None:
        v = grid.reshape(L, L)
        finite = v[np.isfinite(v)]
        if finite[np.argmax(np.abs(finite))] < 0:   # sign: largest cell positive
            v = -v
        amax = float(np.nanmax(np.abs(v)))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.85")                          # excluded origin → gray
        im = ax.imshow(np.ma.masked_invalid(v), origin="upper", cmap=cmap,
                       vmin=-amax, vmax=amax, interpolation="nearest")
        plt.colorbar(im, ax=ax, pad=0.02).set_label(rf"{pc_lab} loading")
        ax.set_xticks([0, L // 2, L - 1])
        ax.set_yticks([0, L // 2, L - 1])
        if is_q:
            ax.set_xlabel(r"$i_y$ \quad ($q_y = 2\pi i_y/L$)")
            ax.set_ylabel(r"$i_x$ \quad ($q_x = 2\pi i_x/L$)")
            ax.set_xticklabels(["0", r"$\pi$", r"$2\pi(L{-}1)/L$"])
            ax.set_yticklabels(["0", r"$\pi$", r"$2\pi(L{-}1)/L$"])
        else:
            ax.set_xlabel(r"$i_y$")
            ax.set_ylabel(r"$i_x$")
        ax.set_title(f"{label} --- {pc_lab} loading on the "
                     + ("Brillouin zone" if is_q else "lattice"),
                     fontsize=10)
    else:
        v = loading
        if v.flat[np.argmax(np.abs(v))] < 0:
            v = -v
        ax.bar(np.arange(len(v)), v, color=_D_SCATTER, width=1.0)
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_xlabel(r"feature index $i$")
        ax.set_ylabel(rf"{pc_lab} loading $w_i$")
        ax.set_title(f"{label} --- {pc_lab} loading vs feature index "
                     rf"($n_{{\mathrm{{feat}}}}={len(v)}$)",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")


# Backwards-compat alias (older callers still expect _diag_panel_pc1_loading)
_diag_panel_pc1_loading = _diag_panel_pc_loading


def _diag_panel_p1_ridgeline(ax, P1_sample, betas_sample, u_beta, betac, label):
    """Stacked per-β KDE of P1 (ridgeline plot)."""
    if betac is not None and np.isfinite(betac):
        norm = TwoSlopeNorm(vcenter=float(betac),
                            vmin=float(u_beta.min()),
                            vmax=float(u_beta.max()))
    else:
        norm = plt.Normalize(vmin=float(u_beta.min()),
                             vmax=float(u_beta.max()))
    cmap = plt.get_cmap("RdBu")

    lo = np.percentile(P1_sample, 1)
    hi = np.percentile(P1_sample, 99)
    pad = 0.05 * (hi - lo)
    x = np.linspace(lo - pad, hi + pad, 400)

    for i, b in enumerate(u_beta):
        s = P1_sample[betas_sample == b]
        if len(s) < 5:
            continue
        try:
            kde  = gaussian_kde(s)
            dens = kde(x)
        except np.linalg.LinAlgError:
            continue
        dens  = dens / dens.max() * 0.85
        color = cmap(norm(b))
        ax.fill_between(x, i, i + dens, alpha=0.6, color=color, lw=0)
        ax.plot(x, i + dens, color=color, lw=0.6)

    if betac is not None and np.isfinite(betac):
        bc_idx = float(np.argmin(np.abs(u_beta - float(betac))))
        ax.axhline(bc_idx, color="red", ls=":", lw=1.0)
        ax.text(x[-1], bc_idx, rf" $\beta_c$", color="red", fontsize=9,
                va="center")

    ax.set_yticks(range(len(u_beta)))
    ax.set_yticklabels([f"{b:g}" for b in u_beta], fontsize=7)
    ax.set_xlabel(r"$P_1$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"{label} --- $P_1$ distribution per $\\beta$",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)


def _diag_panel_spectrum(ax, ev, title):
    """Explained-variance ratio per PC index on a semilog axis."""
    k = np.arange(1, len(ev) + 1)
    ax.plot(k, ev, "o-", lw=1.4, ms=5, color=_D_SCATTER)
    ax.set_yscale("log")
    ax.set_xlabel(r"PC index $k$")
    ax.set_ylabel(r"$\lambda_k / \sum_j \lambda_j$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.annotate(rf"$\lambda_1/\sum = {ev[0]:.3f}$",
                xy=(1, ev[0]), xytext=(1.6, ev[0] * 0.4),
                fontsize=9, color=_D_SCATTER,
                arrowprops=dict(arrowstyle="-", color=_D_SCATTER, lw=0.6))
    if len(ev) >= 2 and ev[1] > 0:
        gap = ev[0] / ev[1]
        ax.annotate(rf"$\lambda_1/\lambda_2 \approx {gap:.0f}$",
                    xy=(2, ev[1]), xytext=(3.5, ev[1] * 4),
                    fontsize=9, color=_D_P2,
                    arrowprops=dict(arrowstyle="-", color=_D_P2, lw=0.6))


# ── Prediction-based panels (Dawid 2022 §3.4.3, Fig. 3.7 c,d) ─────────────────

def _diag_panel_pred_vs_beta(ax, betas_sample, pred_sample, u_beta,
                             mean_pred, std_pred, betac, title):
    """Per-sample (β_true, β_pred) scatter + per-β mean ± std band + identity
    line.  Mirrors :func:`_diag_panel_pc_vs_beta` but with β_pred on the y-axis,
    plus an explicit y = x reference so saturation inside each phase is visible.
    """
    ax.scatter(betas_sample, pred_sample, s=2, alpha=0.15, color=_D_SCATTER,
               rasterized=True, label=r"per-sample $\beta_{\mathrm{pred}}$")
    ax.plot(u_beta, mean_pred, "o-", color=_D_P1, lw=1.6, ms=4,
            label=r"$\langle\beta_{\mathrm{pred}}\rangle_\beta\ \pm\ \mathrm{std}$")
    ax.fill_between(u_beta, mean_pred - std_pred, mean_pred + std_pred,
                    alpha=0.25, color=_D_P1)
    bmin, bmax = float(u_beta[0]), float(u_beta[-1])
    ax.plot([bmin, bmax], [bmin, bmax], "--", lw=0.9, color="black", alpha=0.55,
            label=r"$\beta_{\mathrm{pred}}=\beta$")
    _betac_vline(ax, betac)
    ax.set_xlabel(r"$\beta_{\mathrm{true}}$")
    ax.set_ylabel(r"$\beta_{\mathrm{pred}}$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _diag_panel_pred_curve(ax, u_beta, mean_pred, betac, title):
    """Fig. 3.7(c) — ⟨β_pred⟩_β vs β_true with identity line.

    For a system with no transition the curve is linear; a transition shows up
    as a sigmoidal flattening on either side and a steep central rise.
    """
    bmin, bmax = float(u_beta[0]), float(u_beta[-1])
    ax.plot([bmin, bmax], [bmin, bmax], "--", lw=0.9, color="black", alpha=0.55,
            label=r"$\beta_{\mathrm{pred}}=\beta$")
    ax.plot(u_beta, mean_pred, "o-", color=_D_P1, lw=1.6, ms=5,
            label=r"$\langle\beta_{\mathrm{pred}}\rangle_\beta$")
    _betac_vline(ax, betac)
    ax.set_xlabel(r"$\beta_{\mathrm{true}}$")
    ax.set_ylabel(r"$\langle\beta_{\mathrm{pred}}\rangle_\beta$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _diag_panel_pred_deriv(ax, deriv_beta, deriv_vals, bc_deriv, betac, title):
    """Fig. 3.7(d) — d⟨β_pred⟩/dβ_true (raw) with the argmax peak marked.

    A constant slope = 1 means the NN tracks β perfectly; a peak flags the
    critical point.  β_c is the raw argmax of |d⟨β_pred⟩/dβ|.
    """
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.55,
               label=r"slope $= 1$ (identity)")
    ax.axhline(0.0, color="black", lw=0.4, alpha=0.3)
    ax.plot(deriv_beta, deriv_vals, "o-", color=_D_P1, lw=1.6, ms=4,
            label=r"$d\langle\beta_{\mathrm{pred}}\rangle/d\beta$")

    # Mark the argmax peak in yellow
    idx     = int(np.argmax(np.abs(deriv_vals)))
    peak_b  = float(deriv_beta[idx])
    peak_v  = float(deriv_vals[idx])
    ax.plot(peak_b, peak_v, "o", ms=11, mfc="yellow", mec="black", zorder=5)
    ax.axvline(peak_b, color=_D_P1, ls="--", lw=1.4, alpha=0.85,
               label=rf"$\beta_c^{{\mathrm{{argmax}}}}={bc_deriv:.2f}$")
    _betac_vline(ax, betac)

    ax.set_xlabel(r"$\beta_{\mathrm{true}}$")
    ax.set_ylabel(r"$d\langle\beta_{\mathrm{pred}}\rangle/d\beta_{\mathrm{true}}$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


_C_SEEDS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd",
            "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def _diag_panel_loss_curves(ax, train_curves, val_curves, best_epochs, *,
                            log_y=True, zoom=False, title="",
                            lr_curves=None):
    """Per-seed train (solid) + val (dashed) loss vs epoch.

    `train_curves`, `val_curves` are (n_seeds, max_epochs) NaN-padded.
    `best_epochs[s]` is annotated with a yellow marker at the best val.
    When `lr_curves` is given, the median LR schedule is overlaid on a
    secondary y-axis (right side) so plateau-then-drop episodes are visible.
    Units: MSE on z-scored β labels.
    """
    n_seeds = train_curves.shape[0]
    plotted = False
    for s in range(n_seeds):
        c = _C_SEEDS[s % len(_C_SEEDS)]
        tr = train_curves[s]
        va = val_curves[s]
        mask = ~np.isnan(tr)
        epochs = np.arange(len(tr))[mask]
        if not epochs.size:
            continue
        ax.plot(epochs, tr[mask], "-",  color=c, lw=1.4, alpha=0.9,
                label=f"seed {s} train")
        ax.plot(epochs, va[mask], "--", color=c, lw=1.4, alpha=0.9,
                label=f"seed {s} val")
        be = int(best_epochs[s])
        if 0 <= be < len(tr) and not np.isnan(tr[be]):
            ax.plot(be, va[be], "o", ms=7, mfc="yellow", mec=c, zorder=5)
        plotted = True

    if log_y:
        ax.set_yscale("log")
    if zoom and plotted:
        last_epoch = max(np.sum(~np.isnan(train_curves[s]))
                         for s in range(n_seeds))
        ax.set_xlim(int(last_epoch * 0.4), last_epoch)

    ax.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.5,
               label=r"$\mathrm{Var}(\beta)$ in $z$-units")
    ax.set_xlabel(r"epoch")
    ax.set_ylabel(r"MSE  ($z$-scored $\beta$)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=2)

    # ── LR overlay (median across seeds) ─────────────────────────────────
    if lr_curves is not None and np.isfinite(lr_curves).any():
        ax2 = ax.twinx()
        with np.errstate(invalid="ignore"):
            lr_med = np.nanmedian(lr_curves, axis=0)
        valid = np.isfinite(lr_med)
        if valid.any():
            epochs_all = np.arange(len(lr_med))
            ax2.plot(epochs_all[valid], lr_med[valid],
                     ":", color="gray", lw=1.2, alpha=0.85)
            ax2.set_yscale("log")
            ax2.set_ylabel(r"learning rate (median)", color="gray", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
            ax2.grid(False)


def _diag_panel_seed_losses(ax, best_train, best_val, title):
    """Per-seed final-best train + val MSE as a grouped bar chart.

    R² on validation = 1 − val/Var(β_z) = 1 − val (since Var of z-scored labels
    is 1 by construction).  Printed inline above each val bar.
    """
    n_seeds = len(best_val)
    x       = np.arange(n_seeds)
    w       = 0.4
    ax.bar(x - w/2, best_train, width=w, color="#1f77b4", label="train (best epoch)")
    ax.bar(x + w/2, best_val,   width=w, color="#d62728", label="val (best)")
    for i, v in enumerate(best_val):
        r2 = 1.0 - v
        ax.text(x[i] + w/2, v, rf" $R^2={r2:.2f}$",
                ha="center", va="bottom", fontsize=8, color="#d62728")
    ax.axhline(np.median(best_val), color="#d62728", ls=":", lw=1.0,
               alpha=0.6, label=rf"median val = {np.median(best_val):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in range(n_seeds)])
    ax.set_ylabel(r"MSE  ($z$-scored $\beta$)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=8)


def _diag_panel_seed_epochs(ax, best_epochs, n_epochs_run, title):
    """Per-seed best epoch (val argmin) vs total epochs trained.  Shows
    whether early-stop is firing well-after the best epoch (long tail of
    patience) or whether we'd benefit from more epochs."""
    n_seeds = len(best_epochs)
    x       = np.arange(n_seeds)
    w       = 0.4
    ax.bar(x - w/2, best_epochs,  width=w, color="#2ca02c", label="best epoch")
    ax.bar(x + w/2, n_epochs_run, width=w, color="#888888", label="total epochs")
    for i in range(n_seeds):
        gap = int(n_epochs_run[i]) - int(best_epochs[i])
        ax.text(x[i], n_epochs_run[i], f"+{gap}",
                ha="center", va="bottom", fontsize=8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in range(n_seeds)])
    ax.set_ylabel(r"epoch")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=8)


def _diag_panel_pred_residual(ax, u_beta, mean_pred, std_pred, betac, title):
    """Residual ⟨β_pred⟩_β − β plotted against β.  In a textbook case this is
    flat-negative inside the disordered phase, crosses zero near β_c, and is
    flat-positive inside the ordered phase — the saturation that Fig. 3.7(c)
    only hints at."""
    residual = mean_pred - u_beta
    ax.plot(u_beta, residual, "o-", color=_D_P1, lw=1.5, ms=4)
    ax.fill_between(u_beta, residual - std_pred, residual + std_pred,
                    alpha=0.25, color=_D_P1)
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.6)
    _betac_vline(ax, betac)
    ax.set_xlabel(r"$\beta_{\mathrm{true}}$")
    ax.set_ylabel(r"$\langle\beta_{\mathrm{pred}}\rangle_\beta - \beta$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


# ── PCA — beta sweep ──────────────────────────────────────────────────────────

def _legacy_pca_pages(pdf_var, pdf_km, alpha, key, data, info, model_info,
                      fss):
    """Render the legacy 3-panel susceptibility and 4-panel KMeans pages
    for one alpha into open PdfPages handles.  Kept around behind
    ``LEGACY_VAR_KM_PDFS`` so the km_sig / sm_argmax_P1 estimator overlay
    stays accessible while the diagnostic-style PDFs become primary."""
    bc_arr  = _safe(data, f"{key}_beta_c")
    u_beta  = _safe(data, f"{key}_u_beta")
    if u_beta is None:
        return

    var_P1   = data[f"{key}_var_P1"]
    var_P1sm = data[f"{key}_var_P1_sm"]
    var_P2   = data[f"{key}_var_P2"]
    var_P2sm = data[f"{key}_var_P2_sm"]
    d_beta1  = _safe(data, f"{key}_var1_deriv_beta")
    d_vals1  = _safe(data, f"{key}_var1_deriv_vals")
    d_vals2  = _safe(data, f"{key}_var2_deriv_vals")

    fss_str  = f"{fss:.3f}" if not np.isnan(fss) else "N/A"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        rf"PCA susceptibility  $|$  $\alpha={alpha:.4f}$  "
        rf"$\beta_c(\mathrm{{FSS}})={fss_str}$"
        + "\n" + info,
        fontsize=10)
    _panel_var(axes[0], u_beta, var_P1, var_P1sm, var_P2, var_P2sm,
               bc_arr=bc_arr, fss=fss if not np.isnan(fss) else None)
    axes[0].set_title(r"$\mathrm{Var}(P_1)$ and $\mathrm{Var}(P_2)$ vs $\beta$")
    axes[1].plot(u_beta, var_P1,   "o", ms=4, color=_C_VAR1, alpha=0.5,
                 label=r"$\mathrm{Var}(P_1)$ raw")
    axes[1].plot(u_beta, var_P1sm, "-", lw=2, color=_C_VAR1,
                 label="smoothed")
    _vlines(axes[1], bc_arr, fss if not np.isnan(fss) else None)
    axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel(r"$\mathrm{Var}(P_1)$")
    axes[1].set_title(r"$\mathrm{Var}(P_1)$ --- estimators")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    if d_beta1 is not None and d_vals2 is not None:
        _panel_deriv(axes[2], d_beta1, d_vals1, d_vals2)
    else:
        axes[2].set_visible(False)
    axes[2].set_title(r"$d\,\mathrm{Var}/d\beta$  "
                      r"(zero-crossing $\approx\beta_c$)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf_var.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    P1_s = _safe(data, f"{key}_P1_sample")
    P2_s = _safe(data, f"{key}_P2_sample")
    b_s  = _safe(data, f"{key}_betas_sample")
    km_s = _safe(data, f"{key}_km_labels_sample")
    f_kv = _safe(data, f"{key}_f_kvbs_km")
    popt = _safe(data, f"{key}_km_sig_popt")
    d_kb = _safe(data, f"{key}_km_deriv_beta")
    d_kv = _safe(data, f"{key}_km_deriv_vals")
    if P1_s is None or f_kv is None:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        rf"PCA KMeans  $|$  $\alpha={alpha:.4f}$  "
        rf"$\beta_c(\mathrm{{FSS}})={fss_str}$" + "\n" + info,
        fontsize=10)
    _panel_scatter(axes[0], P1_s, P2_s, b_s, _CM_BETA, r"$\beta$",
                   title=r"$P_1$ vs $P_2$ --- coloured by $\beta$",
                   xlabel=r"$P_1$", ylabel=r"$P_2$")
    _panel_scatter(axes[1], P1_s, P2_s, km_s.astype(float), _CM_CLUSTER,
                   "cluster", title=r"$P_1$ vs $P_2$ --- $k$-means clusters",
                   xlabel=r"$P_1$", ylabel=r"$P_2$")
    ord_lab = _ordered_label(model_info.get("model"))
    ord_inner = ord_lab.strip("$")
    _panel_f_ordered(axes[2], u_beta, f_kv, popt, bc_arr,
                     fss if not np.isnan(fss) else None,
                     ylabel=ord_lab, idx_sig=2, idx_deriv=4)
    axes[2].set_title(rf"{ord_lab} vs $\beta$  (KMeans sigmoid)")
    if d_kb is not None and d_kv is not None:
        _panel_f_deriv(axes[3], d_kb, d_kv, popt)
    else:
        axes[3].set_visible(False)
    axes[3].set_title(rf"$d{ord_inner}/d\beta$")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf_km.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _style_3d(ax):
    """Common axis labels + viewing angle for the (P1,P2,P3) score cloud."""
    ax.set_xlabel(r"$P_1$"); ax.set_ylabel(r"$P_2$"); ax.set_zlabel(r"$P_3$")
    ax.view_init(elev=22, azim=-60)


def _diag_embed3d(P1, P2, P3, betas, km_labels, betac, obs_label,
                  alpha, betac_str):
    """3D scatter of the (P1,P2,P3) PCA score cloud, shown twice: coloured by
    KMeans cluster (what the clustering produced) and by β (the physical
    parameter).  This is the embedding the clustering actually runs in when
    ``PCA_CLUSTER_DIMS >= 3``, so it shows directly how the split sits in space.
    """
    fig = plt.figure(figsize=(16, 7))

    # Left: coloured by KMeans cluster (ordered = higher-mean-β = blue, to match
    # the 2D cluster panel in embedding.pdf).
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    if km_labels is not None:
        km = km_labels.astype(int)
        if (km == 0).any() and (km == 1).any():
            ordered = 0 if betas[km == 0].mean() > betas[km == 1].mean() else 1
        else:
            ordered = 1
        for lab, color, name in ((ordered, _D_SCATTER, "ordered"),
                                 (1 - ordered, _D_P1, "disordered")):
            m = km == lab
            ax0.scatter(P1[m], P2[m], P3[m], s=4, alpha=0.5, c=color,
                        label=name, depthshade=False, rasterized=True)
        ax0.legend(loc="upper left", fontsize=9)
    ax0.set_title("coloured by KMeans cluster", fontsize=11)
    _style_3d(ax0)

    # Right: coloured by β (same sequential colormap as the 2D panels).
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax1.scatter(P1, P2, P3, c=betas, cmap=_CM_BETA, s=4, alpha=0.5,
                     depthshade=False, rasterized=True)
    fig.colorbar(sc, ax=ax1, pad=0.1, fraction=0.03).set_label(r"$\beta$")
    ax1.set_title(r"coloured by $\beta$", fontsize=11)
    _style_3d(ax1)

    fig.suptitle(rf"{obs_label} --- 3D PCA embedding $(P_1,P_2,P_3)$  $|$  "
                 rf"$\alpha={alpha:.4f}$  $\beta_c(\mathrm{{FSS}})={betac_str}$",
                 fontsize=13)
    return fig


def plot_pca_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir,
                        fss_band_dict=None):
    """
    Three multi-page PDFs (one page per alpha) for PCA beta-sweep results:
      avg_analysis.pdf  — 4×1 column: P1 vs β, P2 vs β, ⟨Pk⟩/σk overlay
                          with crossings, Var(Pk|β)/λk overlay with argmax.
      embedding.pdf     — 2×2: (P1,P2) by β, (P1,P2) by KMeans cluster,
                          f_ord(β) sigmoid, df_ord/dβ.
      PC_layout.pdf     — 3 stacked rows: PC1 loading on lattice / BZ,
                          per-β P1 ridgeline KDE, eigenvalue spectrum.

    When ``LEGACY_VAR_KM_PDFS`` is True, also emits the historical
    pca_susceptibility.pdf and pca_kmeans.pdf.

    The run tag is encoded by ``fig_dir`` (`<model>/pca/<tag>/`), so the
    PDF filenames stay clean and content-named.
    """
    if not os.path.exists(npz_path):
        print(f"  [PCA plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    info      = _info_str(model_info)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")
    is_q      = "momentum" in (model_info.get("observable") or "")
    L         = int(model_info.get("L") or 1)

    out_avg = os.path.join(fig_dir, "avg_analysis.pdf")
    out_emb = os.path.join(fig_dir, "embedding.pdf")
    out_pc  = os.path.join(fig_dir, "PC_layout.pdf")

    # 3D embedding (P1,P2,P3) — only when the npz carries the 3rd-PC sample
    # (older results predate it).  Opened/closed outside the `with` so an
    # absent P3 never leaves an empty PDF behind.
    has_p3    = any(f"a{a:.4f}_P3_sample" in data.files for a in alphas)
    out_emb3d = os.path.join(fig_dir, "embedding_3d.pdf")
    pdf_emb3d = PdfPages(out_emb3d) if has_p3 else None

    _BETAC_BANDS.clear()
    if fss_band_dict:
        for a in alphas:
            fss_a = float(fss_dict.get(a, np.nan))
            if not np.isnan(fss_a) and a in fss_band_dict:
                _BETAC_BANDS[float(fss_a)] = fss_band_dict[a]

    legacy_pdfs = []
    if LEGACY_VAR_KM_PDFS:
        out_var = os.path.join(fig_dir, "pca_susceptibility.pdf")
        out_km  = os.path.join(fig_dir, "pca_kmeans.pdf")
        pdf_var = PdfPages(out_var)
        pdf_km  = PdfPages(out_km)
        legacy_pdfs = [(pdf_var, out_var), (pdf_km, out_km)]

    with PdfPages(out_avg) as pdf_avg, \
         PdfPages(out_emb) as pdf_emb, \
         PdfPages(out_pc)  as pdf_pc:
        for alpha in alphas:
            key = f"a{alpha:.4f}"
            fss = float(fss_dict.get(alpha, np.nan))
            betac = fss if not np.isnan(fss) else None
            betac_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"

            u_beta = _safe(data, f"{key}_u_beta")
            if u_beta is None:
                continue
            P1_s    = _safe(data, f"{key}_P1_sample")
            P2_s    = _safe(data, f"{key}_P2_sample")
            P3_s    = _safe(data, f"{key}_P3_sample")
            b_s     = _safe(data, f"{key}_betas_sample")
            km_s    = _safe(data, f"{key}_km_labels_sample")
            var_P1  = _safe(data, f"{key}_var_P1")
            var_P2  = _safe(data, f"{key}_var_P2")
            ev      = _safe(data, f"{key}_explained_var")
            loading   = _safe(data, f"{key}_pc1_loading")
            loading_2 = _safe(data, f"{key}_pc2_loading")
            f_kv    = _safe(data, f"{key}_f_kvbs_km")
            popt    = _safe(data, f"{key}_km_sig_popt")
            d_kb    = _safe(data, f"{key}_km_deriv_beta")
            d_kv    = _safe(data, f"{key}_km_deriv_vals")
            bc_arr  = _safe(data, f"{key}_beta_c")

            if P1_s is None:
                continue

            # ── avg_analysis.pdf  (4×1 column) ────────────────────────────
            fig, axes = plt.subplots(4, 1, figsize=(7.5, 16))
            _diag_panel_pc_vs_beta(
                axes[0], b_s, P1_s, u_beta, betac, r"$P_1$",
                rf"{obs_label} --- $P_1$ vs $\beta$")
            _diag_panel_pc_vs_beta(
                axes[1], b_s, P2_s, u_beta, betac, r"$P_2$",
                rf"{obs_label} --- $P_2$ vs $\beta$")
            _diag_panel_pc_overlay(
                axes[2], P1_s, P2_s, b_s, u_beta, betac,
                rf"{obs_label} --- means of $P_1, P_2$ in std units")
            _diag_panel_pc_var_overlay(
                axes[3], var_P1, var_P2, P1_s, P2_s, u_beta, betac,
                rf"{obs_label} --- variances of $P_1, P_2$ "
                r"(susceptibility, $/\lambda_k$)")
            fig.suptitle(
                rf"PCA averaged analysis --- $\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf_avg.savefig(fig, dpi=120)
            plt.close(fig)

            # ── embedding.pdf  (2×3 layout) ───────────────────────────────
            fig, axes = plt.subplots(2, 3, figsize=(20, 11),
                                     constrained_layout=True)

            # Col 0+1, top row: embedding space, two views of the same data
            _diag_panel_p1p2_scatter(axes[0, 0], P1_s, P2_s, b_s, betac)
            if km_s is not None:
                _diag_panel_p1p2_clusters(axes[0, 1], P1_s, P2_s, km_s, b_s)
                # Lock cluster panel to the β-scatter limits so the two
                # views read as a before/after of the same point cloud.
                axes[0, 1].set_xlim(axes[0, 0].get_xlim())
                axes[0, 1].set_ylim(axes[0, 0].get_ylim())
            else:
                axes[0, 1].set_visible(False)

            # Col 0+1, bottom row: cluster fraction and its sharpness, vs β
            ord_lab = _ordered_label(model_info.get("model"))
            if f_kv is not None:
                _diag_panel_f_sigmoid(axes[1, 0], u_beta, f_kv, popt,
                                      betac, ord_lab)
            else:
                axes[1, 0].set_visible(False)

            if d_kb is not None and d_kv is not None:
                _diag_panel_f_deriv(axes[1, 1], d_kb, d_kv, popt,
                                    betac, ord_lab)
            else:
                axes[1, 1].set_visible(False)

            # Col 2: PC1 + PC2 loadings on the lattice / BZ.
            _diag_panel_pc_loading(axes[0, 2], loading,   L, is_q,
                                   obs_label, pc_index=1)
            _diag_panel_pc_loading(axes[1, 2], loading_2, L, is_q,
                                   obs_label, pc_index=2)

            fig.suptitle(
                rf"{obs_label} --- PCA embedding  $|$  "
                rf"$\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=13)
            pdf_emb.savefig(fig, dpi=120)
            plt.close(fig)

            # ── embedding_3d.pdf  (P1,P2,P3 by cluster and by β) ──────────
            if pdf_emb3d is not None and P3_s is not None:
                fig3d = _diag_embed3d(P1_s, P2_s, P3_s, b_s, km_s, betac,
                                      obs_label, alpha, betac_str)
                pdf_emb3d.savefig(fig3d, dpi=120)
                plt.close(fig3d)

            # ── PC_layout.pdf  (3 stacked rows) ───────────────────────────
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))
            _diag_panel_pc1_loading(axes[0], loading, L, is_q, obs_label)
            _diag_panel_p1_ridgeline(axes[1], P1_s, b_s, u_beta, betac,
                                     obs_label)
            if ev is not None:
                _diag_panel_spectrum(axes[2], ev,
                                     rf"{obs_label} --- PCA spectrum")
            else:
                axes[2].set_visible(False)
            fig.suptitle(rf"PC layout --- $\alpha={alpha:.4f}$",
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf_pc.savefig(fig, dpi=120)
            plt.close(fig)

            # ── Legacy susceptibility / kmeans pages (off by default) ─────
            if LEGACY_VAR_KM_PDFS:
                _legacy_pca_pages(pdf_var, pdf_km, alpha, key, data,
                                  info, model_info, fss)

    for pdf, _ in legacy_pdfs:
        pdf.close()
    if pdf_emb3d is not None:
        pdf_emb3d.close()

    print(f"  Saved: {out_avg}")
    print(f"  Saved: {out_emb}")
    if pdf_emb3d is not None:
        print(f"  Saved: {out_emb3d}")
    print(f"  Saved: {out_pc}")
    for _, p in legacy_pdfs:
        print(f"  Saved: {p}")
    return (out_avg, out_emb, out_pc)


# ── UMAP — beta sweep ─────────────────────────────────────────────────────────

def plot_umap_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir,
                         fss_band_dict=None, method_label="UMAP"):
    """
    Multi-page PDF for a 2D-embedding beta-sweep: ``embedding.pdf`` — same
    2×2 diagnostic-style layout as the PCA embedding.  Used for both UMAP
    and t-SNE (they share the npz key layout); ``method_label`` sets the
    axis / title text.
    Run tag is encoded by ``fig_dir`` (`<model>/<method>/<tag>/`).
    """
    if not os.path.exists(npz_path):
        print(f"  [UMAP plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")
    out       = os.path.join(fig_dir, "embedding.pdf")

    _BETAC_BANDS.clear()
    if fss_band_dict:
        for a in alphas:
            fss_a = float(fss_dict.get(a, np.nan))
            if not np.isnan(fss_a) and a in fss_band_dict:
                _BETAC_BANDS[float(fss_a)] = fss_band_dict[a]

    with PdfPages(out) as pdf:
        for alpha in alphas:
            key       = f"a{alpha:.4f}"
            fss       = float(fss_dict.get(alpha, np.nan))
            betac     = fss if not np.isnan(fss) else None
            betac_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"

            u_beta = _safe(data, f"{key}_u_beta")
            if u_beta is None:
                continue

            emb_x  = _safe(data, f"{key}_emb_x")
            emb_y  = _safe(data, f"{key}_emb_y")
            emb_b  = _safe(data, f"{key}_emb_betas")
            emb_km = _safe(data, f"{key}_emb_km")
            f_kv   = _safe(data, f"{key}_f_kvbs_km")
            popt   = _safe(data, f"{key}_km_sig_popt")
            d_kb   = _safe(data, f"{key}_km_deriv_beta")
            d_kv   = _safe(data, f"{key}_km_deriv_vals")

            if emb_x is None or f_kv is None:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                                     constrained_layout=True)

            # Top row: embedding by β / by cluster
            _diag_panel_p1p2_scatter(
                axes[0, 0], emb_x, emb_y, emb_b, betac,
                xlabel=rf"{method_label}$_1$", ylabel=rf"{method_label}$_2$",
                title=rf"{method_label} --- coloured by $\beta$")
            if emb_km is not None:
                _diag_panel_p1p2_clusters(
                    axes[0, 1], emb_x, emb_y, emb_km, emb_b,
                    xlabel=rf"{method_label}$_1$", ylabel=rf"{method_label}$_2$",
                    title=rf"{method_label} --- coloured by cluster")
                axes[0, 1].set_xlim(axes[0, 0].get_xlim())
                axes[0, 1].set_ylim(axes[0, 0].get_ylim())
            else:
                axes[0, 1].set_visible(False)

            # Bottom row: cluster fraction and its sharpness, vs β
            ord_lab = _ordered_label(model_info.get("model"))
            _diag_panel_f_sigmoid(axes[1, 0], u_beta, f_kv, popt,
                                  betac, ord_lab)

            if d_kb is not None and d_kv is not None:
                _diag_panel_f_deriv(axes[1, 1], d_kb, d_kv, popt,
                                    betac, ord_lab)
            else:
                axes[1, 1].set_visible(False)

            fig.suptitle(
                rf"{obs_label} --- {method_label} embedding  $|$  "
                rf"$\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=13)
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


# ── Prediction-based — beta sweep ─────────────────────────────────────────────

def plot_prediction_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir,
                               fss_band_dict=None):
    """Two PDFs (one page per alpha) for prediction-based β_c results.

    ``avg_analysis.pdf`` — 4×1 column matching the PCA layout:
      1. (β_true, β_pred) per-sample scatter + per-β mean ± std + identity
      2. ⟨β_pred⟩_β raw + smoothed vs β_true (Fig. 3.7c of Dawid 2022)
      3. d⟨β_pred⟩/dβ raw + smoothed with peaks marked (Fig. 3.7d)
      4. Residual ⟨β_pred⟩ − β (saturation diagnostic)

    ``derivative.pdf`` — single panel reproducing only Fig. 3.7(d) for the
    quick-look comparison against the FSS band / other methods.
    """
    if not os.path.exists(npz_path):
        print(f"  [Prediction plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")

    out_avg   = os.path.join(fig_dir, "avg_analysis.pdf")
    out_deriv = os.path.join(fig_dir, "derivative.pdf")
    out_conv  = os.path.join(fig_dir, "convergence.pdf")
    has_conv  = any(f"a{a:.4f}_train_curves" in data.files for a in alphas)

    _BETAC_BANDS.clear()
    if fss_band_dict:
        for a in alphas:
            fss_a = float(fss_dict.get(a, np.nan))
            if not np.isnan(fss_a) and a in fss_band_dict:
                _BETAC_BANDS[float(fss_a)] = fss_band_dict[a]

    pdf_conv = PdfPages(out_conv) if has_conv else None
    with PdfPages(out_avg) as pdf_avg, PdfPages(out_deriv) as pdf_deriv:
        for alpha in alphas:
            key = f"a{alpha:.4f}"
            fss = float(fss_dict.get(alpha, np.nan))
            betac = fss if not np.isnan(fss) else None
            betac_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"

            u_beta     = _safe(data, f"{key}_u_beta")
            mean_pred  = _safe(data, f"{key}_mean_pred")
            std_pred   = _safe(data, f"{key}_std_pred")
            deriv_beta = _safe(data, f"{key}_deriv_beta")
            deriv_vals = _safe(data, f"{key}_deriv_vals")
            bc_arr     = _safe(data, f"{key}_beta_c")
            pred_s     = _safe(data, f"{key}_pred_sample")
            b_s        = _safe(data, f"{key}_betas_sample")

            if u_beta is None or mean_pred is None:
                continue

            bc_deriv = float(bc_arr[0]) if bc_arr is not None else np.nan

            # ── avg_analysis.pdf  (4×1 column) ────────────────────────────
            fig, axes = plt.subplots(4, 1, figsize=(7.5, 16))

            _diag_panel_pred_vs_beta(
                axes[0], b_s, pred_s, u_beta, mean_pred, std_pred,
                betac,
                rf"{obs_label} --- per-sample $\beta_{{\mathrm{{pred}}}}$ vs $\beta$",
            )
            _diag_panel_pred_curve(
                axes[1], u_beta, mean_pred, betac,
                rf"{obs_label} --- $\langle\beta_{{\mathrm{{pred}}}}\rangle_\beta$ "
                r"(Fig.~3.7c)",
            )
            _diag_panel_pred_deriv(
                axes[2], deriv_beta, deriv_vals, bc_deriv, betac,
                rf"{obs_label} --- $d\langle\beta_{{\mathrm{{pred}}}}\rangle/"
                r"d\beta$ (Fig.~3.7d)",
            )
            _diag_panel_pred_residual(
                axes[3], u_beta, mean_pred, std_pred, betac,
                rf"{obs_label} --- saturation residual "
                r"$\langle\beta_{\mathrm{pred}}\rangle - \beta$",
            )

            fig.suptitle(
                rf"Prediction-based analysis --- $\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$  "
                rf"$\beta_c^{{\mathrm{{argmax}}}}={bc_deriv:.2f}$",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf_avg.savefig(fig, dpi=120)
            plt.close(fig)

            # ── derivative.pdf  (single Fig. 3.7d panel) ──────────────────
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            _diag_panel_pred_deriv(
                ax, deriv_beta, deriv_vals, bc_deriv, betac,
                rf"{obs_label} --- $\alpha={alpha:.4f}$",
            )
            fig.suptitle(
                rf"Prediction-based derivative --- "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf_deriv.savefig(fig, dpi=120)
            plt.close(fig)

            # ── convergence.pdf  (2×2 training diagnostics) ───────────────
            if pdf_conv is not None:
                tr_c   = _safe(data, f"{key}_train_curves")
                va_c   = _safe(data, f"{key}_val_curves")
                lr_c   = _safe(data, f"{key}_lr_curves")
                bv_s   = _safe(data, f"{key}_best_val_per_seed")
                bt_s   = _safe(data, f"{key}_best_train_per_seed")
                be_s   = _safe(data, f"{key}_best_epoch_per_seed")
                nep_s  = _safe(data, f"{key}_n_epochs_per_seed")
                if tr_c is None or va_c is None:
                    continue
                fig, axes = plt.subplots(2, 2, figsize=(13, 9))
                _diag_panel_loss_curves(
                    axes[0, 0], tr_c, va_c, be_s,
                    log_y=True, zoom=False, lr_curves=lr_c,
                    title=rf"{obs_label} --- loss + LR vs epoch (log $y$)",
                )
                _diag_panel_loss_curves(
                    axes[0, 1], tr_c, va_c, be_s,
                    log_y=False, zoom=True, lr_curves=lr_c,
                    title=rf"{obs_label} --- loss + LR vs epoch (linear, zoom)",
                )
                _diag_panel_seed_losses(
                    axes[1, 0], bt_s, bv_s,
                    title=rf"{obs_label} --- best train / val MSE per seed",
                )
                _diag_panel_seed_epochs(
                    axes[1, 1], be_s, nep_s,
                    title=rf"{obs_label} --- best epoch vs total epochs",
                )
                fig.suptitle(
                    rf"Prediction-based training diagnostics --- "
                    rf"$\alpha={alpha:.4f}$",
                    fontsize=12,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf_conv.savefig(fig, dpi=120)
                plt.close(fig)

    if pdf_conv is not None:
        pdf_conv.close()

    print(f"  Saved: {out_avg}")
    print(f"  Saved: {out_deriv}")
    if has_conv:
        print(f"  Saved: {out_conv}")
    return (out_avg, out_deriv)


# ── Mean-based (Arnold 2021) — beta sweep ─────────────────────────────────────

def plot_mean_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir,
                         fss_band_dict=None, predict_npz=None):
    """Single-PDF mean-based diagnostic, one page per alpha.

    ``divergence.pdf`` — Δ(β) curve with raw argmax marked.  When the
    matching ``predict_results_*`` npz is available, overlays the NN
    derivative on a twin axis for direct cross-check.
    """
    if not os.path.exists(npz_path):
        print(f"  [Mean plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")

    out = os.path.join(fig_dir, "divergence.pdf")

    _BETAC_BANDS.clear()
    if fss_band_dict:
        for a in alphas:
            fss_a = float(fss_dict.get(a, np.nan))
            if not np.isnan(fss_a) and a in fss_band_dict:
                _BETAC_BANDS[float(fss_a)] = fss_band_dict[a]

    predict_data = None
    if predict_npz and os.path.exists(predict_npz):
        predict_data = np.load(predict_npz, allow_pickle=False)

    with PdfPages(out) as pdf:
        for alpha in alphas:
            key = f"a{alpha:.4f}"
            fss = float(fss_dict.get(alpha, np.nan))
            betac = fss if not np.isnan(fss) else None
            betac_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"

            beta_mid = _safe(data, f"{key}_beta_mid")
            delta    = _safe(data, f"{key}_delta")
            bc_arr   = _safe(data, f"{key}_beta_c")
            if beta_mid is None or delta is None:
                continue
            bc_mean = float(bc_arr[0]) if bc_arr is not None else np.nan

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.plot(beta_mid, delta, "o-", color=_D_P1, lw=1.7, ms=5,
                    label=r"$\Delta(\beta)=\|\bar x(\beta+\Delta\beta)-\bar x(\beta-\Delta\beta)\|_2$")
            idx = int(np.argmax(delta))
            ax.plot(beta_mid[idx], delta[idx], "o",
                    ms=12, mfc="yellow", mec="black", zorder=5)
            ax.axvline(bc_mean, color=_D_P1, ls="--", lw=1.4, alpha=0.85,
                       label=rf"$\beta_c^{{\mathrm{{mean}}}}={bc_mean:.2f}$")
            _betac_vline(ax, betac)
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel(r"$\Delta(\beta)$  (per-$\beta$ mean drift, $z$-scored)")
            ax.set_title(rf"{obs_label} --- Arnold mean-based divergence",
                         fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

            # Optional NN derivative on twin axis
            if predict_data is not None:
                d_beta_nn = _safe(predict_data, f"{key}_deriv_beta")
                d_vals_nn = _safe(predict_data, f"{key}_deriv_vals")
                bc_nn_arr = _safe(predict_data, f"{key}_beta_c")
                if (d_beta_nn is not None and d_vals_nn is not None
                        and bc_nn_arr is not None):
                    ax2 = ax.twinx()
                    ax2.plot(d_beta_nn, d_vals_nn, "s--", color=_D_P2,
                             lw=1.2, ms=4, alpha=0.85,
                             label=rf"NN $d\langle\beta_{{\mathrm{{pred}}}}\rangle/d\beta$")
                    bc_nn = float(bc_nn_arr[0])
                    ax2.axvline(bc_nn, color=_D_P2, ls=":", lw=1.4, alpha=0.85,
                                label=rf"$\beta_c^{{\mathrm{{NN}}}}={bc_nn:.2f}$")
                    ax2.set_ylabel(r"$d\langle\beta_{\mathrm{pred}}\rangle/d\beta$",
                                   color=_D_P2, fontsize=10)
                    ax2.tick_params(axis="y", labelcolor=_D_P2)
                    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

            fig.suptitle(
                rf"Mean-based --- $\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


def plot_lbc_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir,
                        fss_band_dict=None):
    """Single-PDF Learning-by-Confusion diagnostic, one page per alpha.

    ``confusion.pdf`` — the W-curve: held-out accuracy (mean ± seed std) vs the
    trial β_c', with the interior W peak marked as the LBC β_c estimate and the
    FSS reference β_c drawn as a dotted line.
    """
    if not os.path.exists(npz_path):
        print(f"  [LBC plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")

    out = os.path.join(fig_dir, "confusion.pdf")

    _BETAC_BANDS.clear()
    if fss_band_dict:
        for a in alphas:
            fss_a = float(fss_dict.get(a, np.nan))
            if not np.isnan(fss_a) and a in fss_band_dict:
                _BETAC_BANDS[float(fss_a)] = fss_band_dict[a]

    with PdfPages(out) as pdf:
        for alpha in alphas:
            key = f"a{alpha:.4f}"
            fss = float(fss_dict.get(alpha, np.nan))
            betac = fss if not np.isnan(fss) else None
            betac_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"

            cands    = _safe(data, f"{key}_candidates")
            mean_acc = _safe(data, f"{key}_mean_acc")
            std_acc  = _safe(data, f"{key}_std_acc")
            bc_arr   = _safe(data, f"{key}_beta_c")
            if cands is None or mean_acc is None:
                continue
            bc_lbc = float(bc_arr[0]) if bc_arr is not None else np.nan

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.errorbar(cands, mean_acc, yerr=std_acc,
                        fmt="o-", color=_D_P1, lw=1.7, ms=5,
                        ecolor=_D_P1, elinewidth=0.9, capsize=2,
                        label=r"CV test accuracy $\pm$ seed std")
            ax.axvline(bc_lbc, color=_D_P1, ls="--", lw=1.4, alpha=0.85,
                       label=rf"$\beta_c^{{\mathrm{{LBC}}}}={bc_lbc:.2f}$ (W peak)")
            # highlight the W middle peak
            idx = int(np.argmin(np.abs(cands - bc_lbc)))
            ax.plot(cands[idx], mean_acc[idx], "*",
                    ms=16, mfc="yellow", mec="black", zorder=5)
            _betac_vline(ax, betac)

            ax.set_xlabel(r"trial $\beta_c'$")
            ax.set_ylabel("test accuracy")
            ax.set_ylim(0.45, 1.05)
            ax.set_title(rf"{obs_label} --- Learning-by-Confusion W-curve",
                         fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower center", fontsize=9, framealpha=0.9)

            fig.suptitle(
                rf"Learning by Confusion --- $\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


def plot_lbc_training_curves(npz_path, alphas, model_info, fig_dir):
    """Per-epoch training diagnostic for LBC, one page per alpha.

    ``training_curves.pdf`` — for the representative candidate (the W peak),
    three panels show loss, accuracy, and learning rate vs epoch, with one line
    per seed (train solid, val dashed) and the early-stop best epoch marked.
    Silently does nothing if the npz carries no training history.
    """
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path, allow_pickle=False)
    have = [a for a in alphas if f"a{a:.4f}_hist_val_loss" in data.files]
    if not have:
        return

    obs_label = (model_info.get("observable") or "").replace("_", r"\_")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "training_curves.pdf")

    with PdfPages(out) as pdf:
        for alpha in have:
            key = f"a{alpha:.4f}"
            tr_loss = data[f"{key}_hist_train_loss"]
            va_loss = data[f"{key}_hist_val_loss"]
            tr_acc  = data[f"{key}_hist_train_acc"]
            va_acc  = data[f"{key}_hist_val_acc"]
            lr_hist = data[f"{key}_hist_lr"]
            best_ep = _safe(data, f"{key}_hist_best_epoch")
            bcp_arr = _safe(data, f"{key}_hist_beta_c_prime")
            bcp = float(bcp_arr[0]) if bcp_arr is not None else np.nan

            n_seeds = tr_loss.shape[0]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

            for s in range(n_seeds):
                color = f"C{s}"
                valid = ~np.isnan(va_loss[s])
                ep = np.arange(1, int(valid.sum()) + 1)
                axes[0].plot(ep, tr_loss[s][valid], "-",  color=color,
                             lw=1.3, label=f"seed {s} train")
                axes[0].plot(ep, va_loss[s][valid], "--", color=color,
                             lw=1.3, label=f"seed {s} val")
                axes[1].plot(ep, tr_acc[s][valid], "-",  color=color, lw=1.3)
                axes[1].plot(ep, va_acc[s][valid], "--", color=color, lw=1.3)
                axes[2].plot(ep, lr_hist[s][valid], "-",  color=color, lw=1.3,
                             label=f"seed {s}")
                if best_ep is not None and s < len(best_ep):
                    be = int(best_ep[s])
                    axes[0].plot(be + 1, va_loss[s][be], "o", color=color, ms=7,
                                 mec="black", zorder=5)

            axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss")
            axes[0].set_title("loss (— train, -- val; • best epoch)")
            axes[0].legend(fontsize=7, ncol=max(1, n_seeds))
            axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
            axes[1].set_title("accuracy (— train, -- val)")
            axes[2].set_xlabel("epoch"); axes[2].set_ylabel("learning rate")
            axes[2].set_yscale("log")
            axes[2].set_title("learning rate")
            axes[2].legend(fontsize=8)
            for ax in axes:
                ax.grid(alpha=0.3)

            fig.suptitle(
                rf"LBC training --- {obs_label} --- $\alpha={alpha:.4f}$  "
                rf"(representative candidate $\beta_c'={bcp:.2f}$)",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


def _assaad_nskips(data, alpha):
    """nskip values present in an assaad npz for one alpha, sorted ascending."""
    import re
    pat = re.compile(rf"a{alpha:.4f}_ns(\d+)_accs$")
    found = {int(m.group(1)) for f in data.files for m in [pat.match(f)] if m}
    return sorted(found)


def plot_assaad_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir,
                           fss_band_dict=None):
    """Single-PDF Assaad/GI diagnostic, one page per alpha.

    ``accuracy.pdf`` — test accuracy vs guessed β_c, one curve per nskip window
    (overlaid), each curve's peak marked as its β_c estimate, with the FSS
    reference β_c drawn as a dotted line.
    """
    if not os.path.exists(npz_path):
        print(f"  [Assaad plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")

    out = os.path.join(fig_dir, "accuracy.pdf")

    _BETAC_BANDS.clear()
    if fss_band_dict:
        for a in alphas:
            fss_a = float(fss_dict.get(a, np.nan))
            if not np.isnan(fss_a) and a in fss_band_dict:
                _BETAC_BANDS[float(fss_a)] = fss_band_dict[a]

    with PdfPages(out) as pdf:
        for alpha in alphas:
            nskips = _assaad_nskips(data, alpha)
            if not nskips:
                continue
            fss = float(fss_dict.get(alpha, np.nan))
            betac = fss if not np.isnan(fss) else None
            betac_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"

            fig, ax = plt.subplots(figsize=(9, 5))
            for j, nskip in enumerate(nskips):
                key = f"a{alpha:.4f}_ns{nskip}"
                betas = _safe(data, f"{key}_betas")
                accs  = _safe(data, f"{key}_accs")
                bc_arr = _safe(data, f"{key}_beta_c")
                if betas is None or accs is None:
                    continue
                color = f"C{j}"
                bc = float(bc_arr[0]) if bc_arr is not None else np.nan
                accs_std = _safe(data, f"{key}_accs_std")
                ax.errorbar(betas, accs, yerr=accs_std, fmt="o-", color=color,
                            ecolor=color, elinewidth=0.8, capsize=2, lw=1.6, ms=5,
                            label=rf"nskip$={nskip}$, $\beta_c={bc:.2f}$")
                idx = int(np.argmax(accs))
                ax.plot(betas[idx], accs[idx], "*",
                        ms=15, mfc="yellow", mec="black", zorder=5)
            _betac_vline(ax, betac)

            ax.set_xlabel(r"guessed $\beta_c$")
            ax.set_ylabel("test accuracy")
            ax.set_title(rf"{obs_label} --- Assaad/GI peak accuracy",
                         fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

            fig.suptitle(
                rf"Assaad/GI --- $\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


def plot_assaad_training_curves(npz_path, alphas, model_info, fig_dir):
    """Per-epoch training diagnostic for Assaad, one page per (alpha, nskip).

    ``training_curves.pdf`` — for the representative guess (closest to FSS β_c)
    of each nskip, three panels show loss, accuracy, and learning rate vs epoch
    (train solid, val dashed), with the early-stop best epoch marked.  Silently
    does nothing if the npz carries no training history.
    """
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "training_curves.pdf")

    pages = 0
    with PdfPages(out) as pdf:
        for alpha in alphas:
            for nskip in _assaad_nskips(data, alpha):
                key = f"a{alpha:.4f}_ns{nskip}"
                vl = _safe(data, f"{key}_hist_val_loss")
                if vl is None:
                    continue
                tl  = data[f"{key}_hist_train_loss"]
                ta  = data[f"{key}_hist_train_acc"]
                va  = data[f"{key}_hist_val_acc"]
                lr  = data[f"{key}_hist_lr"]
                be_arr  = _safe(data, f"{key}_hist_best_epoch")
                bcp_arr = _safe(data, f"{key}_hist_beta_c_prime")
                bcp = float(bcp_arr[0]) if bcp_arr is not None else np.nan

                n_seeds = vl.shape[0]
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
                for s in range(n_seeds):
                    color = f"C{s}"
                    valid = ~np.isnan(vl[s])
                    ep = np.arange(1, int(valid.sum()) + 1)
                    axes[0].plot(ep, tl[s][valid], "-",  color=color, lw=1.3,
                                 label=f"seed {s} train")
                    axes[0].plot(ep, vl[s][valid], "--", color=color, lw=1.3,
                                 label=f"seed {s} val")
                    axes[1].plot(ep, ta[s][valid], "-",  color=color, lw=1.3)
                    axes[1].plot(ep, va[s][valid], "--", color=color, lw=1.3)
                    axes[2].plot(ep, lr[s][valid], "-",  color=color, lw=1.3,
                                 label=f"seed {s}")
                    if be_arr is not None and s < len(be_arr):
                        be = int(be_arr[s])
                        axes[0].plot(be + 1, vl[s][be], "o", color=color, ms=7,
                                     mec="black", zorder=5)

                axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
                axes[0].set_title("loss (— train, -- val; • best epoch)")
                axes[0].legend(fontsize=7, ncol=max(1, n_seeds))
                axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
                axes[1].set_title("accuracy (— train, -- val)")
                axes[2].set_xlabel("epoch"); axes[2].set_ylabel("learning rate")
                axes[2].set_yscale("log")
                axes[2].set_title("learning rate")
                axes[2].legend(fontsize=8)
                for ax in axes:
                    ax.grid(alpha=0.3)

                fig.suptitle(
                    rf"Assaad training --- {obs_label} --- $\alpha={alpha:.4f}$, "
                    rf"nskip$={nskip}$  (representative guess $\beta_c={bcp:.2f}$)",
                    fontsize=12,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.94])
                pdf.savefig(fig, dpi=120)
                plt.close(fig)
                pages += 1

    if pages:
        print(f"  Saved: {out}")
    return out


# ── Summary: β_c vs α ────────────────────────────────────────────────────────

def plot_beta_c_summary(pca_npz, umap_npz, alphas, fss_dict, model_info, fig_dir):
    """
    Single figure: β_c vs α, overlaying PCA, UMAP, and FSS estimates.
    Run tag is encoded by ``fig_dir`` (`<model>/summary/<tag>/`).
    """
    out   = os.path.join(fig_dir, "beta_c_summary.pdf")

    pca_data  = np.load(pca_npz,  allow_pickle=False) if (pca_npz  and os.path.exists(pca_npz))  else None
    umap_data = np.load(umap_npz, allow_pickle=False) if (umap_npz and os.path.exists(umap_npz)) else None

    pca_sm   = []   # PCA sm_argmax_P1 (idx 5)
    pca_km   = []   # PCA km_sig       (idx 2)
    pca_err  = []
    umap_km  = []   # UMAP km_sig      (idx 0)
    umap_err = []
    fss_vals = []

    for alpha in alphas:
        key  = f"a{alpha:.4f}"
        fss  = float(fss_dict.get(alpha, np.nan))
        fss_vals.append(fss)

        if pca_data is not None and f"{key}_beta_c" in pca_data:
            bc = pca_data[f"{key}_beta_c"]
            pca_sm.append(bc[5])
            pca_km.append(bc[2])
            pca_err.append(bc[3])
        else:
            pca_sm.append(np.nan); pca_km.append(np.nan); pca_err.append(np.nan)

        if umap_data is not None and f"{key}_beta_c" in umap_data:
            bc = umap_data[f"{key}_beta_c"]
            umap_km.append(bc[0])
            umap_err.append(bc[1])
        else:
            umap_km.append(np.nan); umap_err.append(np.nan)

    a  = np.array(alphas)
    fs = np.array(fss_vals)
    ps = np.array(pca_sm);  pk = np.array(pca_km);  pe = np.array(pca_err)
    uk = np.array(umap_km); ue = np.array(umap_err)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(r"$\beta_c$ vs $\alpha$ --- all methods" + "\n"
                 + _info_str(model_info), fontsize=10)

    fss_mask = ~np.isnan(fs)
    if fss_mask.any():
        ax.plot(a[fss_mask], fs[fss_mask], "ks-", ms=10, lw=2,
                label="FSS reference", zorder=5)

    pca_mask = ~np.isnan(ps)
    if pca_mask.any():
        ax.plot(a[pca_mask], ps[pca_mask], "ro-", ms=7, lw=1.5,
                label=r"PCA smooth argmax $P_1$")
        ax.errorbar(a[pca_mask], pk[pca_mask], yerr=pe[pca_mask],
                    fmt="r^--", ms=6, lw=1.2, capsize=3,
                    label=r"PCA KM sigmoid $\pm 1\sigma$")

    umap_mask = ~np.isnan(uk)
    if umap_mask.any():
        ax.errorbar(a[umap_mask], uk[umap_mask], yerr=ue[umap_mask],
                    fmt="gv-.", ms=7, lw=1.5, capsize=3,
                    label=r"UMAP KM sigmoid $\pm 1\sigma$")

    ax.set_xlabel(r"$\alpha$ (e-ph coupling)")
    ax.set_ylabel(r"$\beta_c$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── PCA — alpha sweep ─────────────────────────────────────────────────────────

def plot_pca_alpha_sweep(npz_path, betas, model_info, fig_dir):
    """
    Multi-page PDF for PCA alpha-sweep (one page per beta value).
    Panels: scatter by α | scatter by cluster | Var(α) | f_CDW
    """
    if not os.path.exists(npz_path):
        print(f"  [PCA alpha-sweep plot] Not found: {npz_path}"); return
    data   = np.load(npz_path, allow_pickle=False)
    info   = _info_str(model_info)
    out    = os.path.join(fig_dir, "pca_alpha_sweep.pdf")

    with PdfPages(out) as pdf:
        for beta in betas:
            key    = f"b{beta:.2f}"
            ac_arr = _safe(data, f"{key}_alpha_c")
            u_alpha = _safe(data, f"{key}_u_alpha")
            if u_alpha is None:
                continue

            var_P1   = _safe(data, f"{key}_var_P1")
            var_P1sm = _safe(data, f"{key}_var_P1_sm")
            var_P2   = _safe(data, f"{key}_var_P2")
            var_P2sm = _safe(data, f"{key}_var_P2_sm")
            P1_s     = _safe(data, f"{key}_P1_sample")
            P2_s     = _safe(data, f"{key}_P2_sample")
            a_s      = _safe(data, f"{key}_alphas_sample")
            km_s     = _safe(data, f"{key}_km_labels_sample")
            f_cdw    = _safe(data, f"{key}_f_cdw_km")
            popt     = _safe(data, f"{key}_km_sig_popt")
            d_ab     = _safe(data, f"{key}_km_deriv_alpha")
            d_av     = _safe(data, f"{key}_km_deriv_vals")

            ac_str = ""
            if ac_arr is not None:
                ac_str = rf"  $|$  $\alpha_c$(KM sig)$={ac_arr[2]:.3f}$"

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(
                rf"PCA $\alpha$-sweep  $|$  $\beta={beta:.2f}${ac_str}"
                + "\n" + info,
                fontsize=10)

            if P1_s is not None:
                _panel_scatter(axes[0], P1_s, P2_s, a_s, _CM_BETA, r"$\alpha$",
                               title=r"$P_1$ vs $P_2$ --- coloured by $\alpha$",
                               xlabel=r"$P_1$", ylabel=r"$P_2$")

                _panel_scatter(axes[1], P1_s, P2_s,
                               km_s.astype(float), _CM_CLUSTER, "cluster",
                               title=r"$P_1$ vs $P_2$ --- $k$-means clusters",
                               xlabel=r"$P_1$", ylabel=r"$P_2$")
            else:
                axes[0].set_visible(False); axes[1].set_visible(False)

            if var_P1 is not None:
                _panel_var(axes[2], u_alpha, var_P1, var_P1sm, var_P2, var_P2sm,
                           bc_arr=ac_arr, xlabel=r"$\alpha$")
                axes[2].set_title(r"$\mathrm{Var}(P_1)$ and $\mathrm{Var}(P_2)$ vs $\alpha$")

            if f_cdw is not None:
                _panel_f_ordered(axes[3], u_alpha, f_cdw, popt, ac_arr, None,
                                 xlabel=r"$\alpha$", ylabel=r"$f_\mathrm{CDW}$",
                                 idx_sig=2, idx_deriv=4)
                axes[3].set_title(r"$f_\mathrm{CDW}$ vs $\alpha$  (KMeans sigmoid)")

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


# ── UMAP — alpha sweep ────────────────────────────────────────────────────────

def plot_umap_alpha_sweep(npz_path, betas, model_info, fig_dir):
    """
    Multi-page PDF for UMAP alpha-sweep (one page per beta value).
    Panels: embedding by α | embedding by cluster | f_CDW | df_CDW/dα
    """
    if not os.path.exists(npz_path):
        print(f"  [UMAP alpha-sweep plot] Not found: {npz_path}"); return
    data   = np.load(npz_path, allow_pickle=False)
    info   = _info_str(model_info)
    out    = os.path.join(fig_dir, "umap_alpha_sweep.pdf")

    with PdfPages(out) as pdf:
        for beta in betas:
            key     = f"b{beta:.2f}"
            ac_arr  = _safe(data, f"{key}_alpha_c")
            u_alpha = _safe(data, f"{key}_u_alpha")
            if u_alpha is None:
                continue

            emb_x  = _safe(data, f"{key}_emb_x")
            emb_y  = _safe(data, f"{key}_emb_y")
            emb_a  = _safe(data, f"{key}_emb_alphas")
            emb_km = _safe(data, f"{key}_emb_km")
            f_cdw  = _safe(data, f"{key}_f_cdw_km")
            popt   = _safe(data, f"{key}_km_sig_popt")
            d_ab   = _safe(data, f"{key}_km_deriv_alpha")
            d_av   = _safe(data, f"{key}_km_deriv_vals")

            ac_str = ""
            if ac_arr is not None:
                ac_str = rf"  $|$  $\alpha_c$(KM sig)$={ac_arr[0]:.3f}$"

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(
                rf"UMAP $\alpha$-sweep  $|$  $\beta={beta:.2f}${ac_str}"
                + "\n" + info,
                fontsize=10)

            if emb_x is not None:
                _panel_scatter(axes[0], emb_x, emb_y, emb_a, _CM_BETA, r"$\alpha$",
                               title=r"UMAP --- coloured by $\alpha$",
                               xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$")

                _panel_scatter(axes[1], emb_x, emb_y,
                               emb_km.astype(float), _CM_CLUSTER, "cluster",
                               title=r"UMAP --- $k$-means clusters",
                               xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$")
            else:
                axes[0].set_visible(False); axes[1].set_visible(False)

            if f_cdw is not None:
                # UMAP alpha_c layout: [km_sig(0), km_sig_err(1), km_deriv(2), best(3)]
                _panel_f_ordered(axes[2], u_alpha, f_cdw, popt, ac_arr, None,
                                 xlabel=r"$\alpha$", ylabel=r"$f_\mathrm{CDW}$",
                                 idx_sm=None, idx_sig=0, idx_deriv=2)
                axes[2].set_title(r"$f_\mathrm{CDW}$ vs $\alpha$  (UMAP $k$-means)")

            if d_ab is not None and d_av is not None:
                _panel_f_deriv(axes[3], d_ab, d_av, popt, xlabel=r"$\alpha$")
            else:
                axes[3].set_visible(False)
            axes[3].set_title(r"$df_\mathrm{CDW}/d\alpha$")

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


# ── Top-level orchestrator ────────────────────────────────────────────────────

def plot_input_lattices(npz_path, alphas, fig_dir, model_info):
    """Provenance page(s): the actual input lattices fed to this ML method.

    Reads the ``a{alpha}_inlat_grids`` / ``..._inlat_betas`` snapshots that the
    method saved (per-β-mean feature maps on the L×L grid) and renders, for each
    alpha, a row of those maps.  This makes every method's PDF self-documenting:
    you can see exactly what observable / transform the method consumed.
    Silently does nothing if the npz carries no snapshots (non-lattice input).
    """
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path, allow_pickle=False)
    have = [a for a in alphas if f"a{a:.4f}_inlat_grids" in data.files]
    if not have:
        return

    obs       = (model_info.get("observable") or "").replace("_", r"\_")
    transform = model_info.get("transform")
    tname     = transform or "raw"
    signed    = transform == "deltasusc"
    cmap      = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("0.85")                       # masked origin renders gray
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "input_lattices.pdf")

    with PdfPages(out) as pdf:
        for alpha in have:
            grids = data[f"a{alpha:.4f}_inlat_grids"]
            sbeta = data[f"a{alpha:.4f}_inlat_betas"]
            ncol  = len(grids)
            # shared color limits across the row so the β-growth is visible
            # (the maps are already globally ÷max|dev| normalized).
            finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
            if signed:
                vmax = float(np.abs(finite).max()) or 1.0
                vmin = -vmax
            else:
                vmin, vmax = float(finite.min()), float(finite.max())
            fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.5),
                                     squeeze=False)
            im = None
            for ax, b, g in zip(axes[0], sbeta, grids):
                im = ax.imshow(np.ma.masked_invalid(g), origin="lower",
                               cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(rf"$\beta t = {b:g}$", fontsize=9)
                ax.set_xlabel(r"$r_1$"); ax.set_ylabel(r"$r_2$")
                ax.set_xticks([0, 3, 6, 9]); ax.set_yticks([0, 3, 6, 9])
            fig.colorbar(im, ax=list(axes[0]), fraction=0.046, pad=0.04)
            fig.suptitle(
                rf"ML input --- {obs} --- transform: {tname} --- "
                rf"$\alpha={alpha:.4f}$"
                "\n" r"(per-$\beta$ mean of the feature vectors fed to the method;"
                r" shared color scale)",
                fontsize=10)
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
    print(f"  Saved: {out}")


def run_all_plots(tag, alphas, fss_dict, model_info,
                  alpha_sweep=False, betas=None,
                  skip_umap=False, results_dir=None,
                  fss_band_dict=None):
    """
    Locate .npz result files by tag and generate all PDFs.

    Parameters
    ----------
    tag          : str or None — run tag (used to find .npz files and name PDFs)
    alphas       : list of float — for beta-sweep mode
    fss_dict     : dict mapping alpha → FSS β_c
    model_info   : dict with keys model, observable, L, beta_min, beta_max, tag
    alpha_sweep  : bool — True for alpha-sweep mode
    betas        : list of float — fixed beta values (alpha-sweep mode)
    skip_umap    : bool — skip UMAP plots even if .npz exists
    results_dir  : str or None — override RESULTS_DIR
    """
    rd        = results_dir or _cfg.RESULTS_DIR
    npz_dir   = os.path.join(rd, "npz")
    model     = model_info.get("model") or "unspecified"
    pca_dir   = _figures_dir(tag, "pca",     model, rd)
    umap_dir  = _figures_dir(tag, "umap",    model, rd)
    summ_dir  = _figures_dir(tag, "summary", model, rd)
    print(f"\n{'='*60}")
    print(f"Plotting  →  results/figures/{model}/<method>/{tag or '_'}/")
    print(f"{'='*60}")

    if alpha_sweep:
        if not betas:
            print("  [plot] alpha_sweep=True but no betas given — skipping."); return

        pca_npz  = os.path.join(npz_dir, "pca_results_alpha_sweep.npz")
        umap_npz = os.path.join(npz_dir, "umap_results_alpha_sweep.npz")
        if tag:
            # also check tag-specific file
            t_pca  = os.path.join(npz_dir, f"pca_results_{tag}.npz")
            t_umap = os.path.join(npz_dir, f"umap_results_{tag}.npz")
            if os.path.exists(t_pca):  pca_npz  = t_pca
            if os.path.exists(t_umap): umap_npz = t_umap

        print("\n[PCA alpha-sweep]")
        plot_pca_alpha_sweep(pca_npz, betas, model_info, pca_dir)

        if not skip_umap:
            print("\n[UMAP alpha-sweep]")
            plot_umap_alpha_sweep(umap_npz, betas, model_info, umap_dir)

    else:
        t_suf       = f"_{tag}" if tag else ""
        pca_npz     = os.path.join(npz_dir, f"pca_results{t_suf}.npz")
        umap_npz    = os.path.join(npz_dir, f"umap_results{t_suf}.npz")
        tsne_npz    = os.path.join(npz_dir, f"tsne_results{t_suf}.npz")
        predict_npz = os.path.join(npz_dir, f"predict_results{t_suf}.npz")
        mean_npz    = os.path.join(npz_dir, f"mean_results{t_suf}.npz")
        lbc_npz     = os.path.join(npz_dir, f"lbc_results{t_suf}.npz")
        assaad_npz  = os.path.join(npz_dir, f"assaad_results{t_suf}.npz")
        tsne_dir    = _figures_dir(tag, "tsne",       model, rd)
        predict_dir = _figures_dir(tag, "prediction", model, rd)
        mean_dir    = _figures_dir(tag, "mean",       model, rd)
        lbc_dir     = _figures_dir(tag, "lbc",        model, rd)
        assaad_dir  = _figures_dir(tag, "assaad",     model, rd)

        if os.path.exists(pca_npz):
            print("\n[PCA beta-sweep]")
            plot_pca_beta_sweep(pca_npz, alphas, fss_dict, model_info, pca_dir,
                                fss_band_dict=fss_band_dict)

        if not skip_umap and os.path.exists(umap_npz):
            print("\n[UMAP beta-sweep]")
            plot_umap_beta_sweep(umap_npz, alphas, fss_dict, model_info, umap_dir,
                                 fss_band_dict=fss_band_dict)

        if os.path.exists(tsne_npz):
            print("\n[t-SNE beta-sweep]")
            plot_umap_beta_sweep(tsne_npz, alphas, fss_dict, model_info, tsne_dir,
                                 fss_band_dict=fss_band_dict, method_label="t-SNE")

        if os.path.exists(predict_npz):
            print("\n[Prediction beta-sweep]")
            plot_prediction_beta_sweep(predict_npz, alphas, fss_dict,
                                       model_info, predict_dir,
                                       fss_band_dict=fss_band_dict)

        if os.path.exists(mean_npz):
            print("\n[Mean-based beta-sweep]")
            plot_mean_beta_sweep(mean_npz, alphas, fss_dict,
                                 model_info, mean_dir,
                                 fss_band_dict=fss_band_dict,
                                 predict_npz=(predict_npz
                                              if os.path.exists(predict_npz)
                                              else None))

        if os.path.exists(lbc_npz):
            print("\n[Learning-by-Confusion beta-sweep]")
            plot_lbc_beta_sweep(lbc_npz, alphas, fss_dict,
                                model_info, lbc_dir,
                                fss_band_dict=fss_band_dict)
            plot_lbc_training_curves(lbc_npz, alphas, model_info, lbc_dir)

        if os.path.exists(assaad_npz):
            print("\n[Assaad (GI) beta-sweep]")
            plot_assaad_beta_sweep(assaad_npz, alphas, fss_dict,
                                   model_info, assaad_dir,
                                   fss_band_dict=fss_band_dict)
            plot_assaad_training_curves(assaad_npz, alphas, model_info, assaad_dir)

        if os.path.exists(pca_npz) and os.path.exists(umap_npz):
            print("\n[Summary: β_c vs α]")
            plot_beta_c_summary(pca_npz, umap_npz, alphas, fss_dict, model_info,
                                summ_dir)

        # Input-lattice provenance: one PDF per method showing the exact maps
        # that method consumed (observable + transform), so every method's
        # output is self-documenting about its input.
        print("\n[Input lattices]")
        for npz_path, fdir in ((pca_npz, pca_dir), (umap_npz, umap_dir),
                               (tsne_npz, tsne_dir), (predict_npz, predict_dir),
                               (mean_npz, mean_dir), (lbc_npz, lbc_dir),
                               (assaad_npz, assaad_dir)):
            plot_input_lattices(npz_path, alphas, fdir, model_info)

    print(f"\nFigures saved under:")
    if os.path.exists(pca_npz):
        print(f"  PCA        →  {pca_dir}")
    if not skip_umap and os.path.exists(umap_npz):
        print(f"  UMAP       →  {umap_dir}")
    if not alpha_sweep and os.path.exists(tsne_npz):
        print(f"  t-SNE      →  {tsne_dir}")
    if os.path.exists(predict_npz):
        print(f"  Prediction →  {predict_dir}")
    if os.path.exists(mean_npz):
        print(f"  Mean       →  {mean_dir}")
    if not alpha_sweep and os.path.exists(lbc_npz):
        print(f"  LBC        →  {lbc_dir}")
    if not alpha_sweep and os.path.exists(assaad_npz):
        print(f"  Assaad     →  {assaad_dir}")
    if not alpha_sweep and os.path.exists(pca_npz) and os.path.exists(umap_npz):
        print(f"  Summary    →  {summ_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ML phase detection results from .npz files")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Model preset (sets alphas, FSS dict, observable)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values to plot (default: all from preset)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag — used to find the correct .npz files")
    parser.add_argument("--file-template", type=str, default=None,
                        help="File template (used to derive observable label)")
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override DATA_DIR in config")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override RESULTS_DIR (where .npz files live)")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Plot alpha-sweep results instead of beta-sweep")
    parser.add_argument("--betas", nargs="+", type=float, default=None,
                        help="Beta value(s) used in the alpha sweep")
    args = parser.parse_args()

    # ── Merge: CLI args override the config block at the top of this file ────
    model       = args.model       or MODEL
    tag         = args.tag         or TAG
    beta_min    = args.beta_min    if args.beta_min    is not None else BETA_MIN
    beta_max    = args.beta_max    if args.beta_max    is not None else BETA_MAX
    alpha_sweep = args.alpha_sweep or ALPHA_SWEEP
    betas       = args.betas       or BETAS
    skip_umap   = args.skip_umap   or SKIP_UMAP
    results_dir = args.results_dir or RESULTS_DIR_OVERRIDE
    alphas_cli  = args.alphas      or ALPHAS   # may still be None → use preset

    if model:
        _cfg.apply_model(model)
    if args.data_dir:
        _cfg.DATA_DIR = args.data_dir

    alphas        = alphas_cli or _cfg.ALPHAS
    file_template = args.file_template or _cfg.FILE_TEMPLATE
    fss_dict      = _cfg.FSS_DICT
    fss_band_dict = getattr(_cfg, "FSS_BAND_DICT", None)

    # Derive observable label from file template
    observable = file_template.split("_L")[0] if file_template else ""
    observable = "_".join(s for s in observable.split("_") if "{" not in s)

    model_info = {
        "model":      model or "ssh",
        "observable": observable,
        "L":          _cfg.L,
        "beta_min":   beta_min,
        "beta_max":   beta_max,
        "tag":        tag,
    }

    run_all_plots(
        tag           = tag,
        alphas        = alphas,
        fss_dict      = fss_dict,
        fss_band_dict = fss_band_dict,
        model_info    = model_info,
        alpha_sweep   = alpha_sweep,
        betas         = betas,
        skip_umap     = skip_umap,
        results_dir   = results_dir,
    )


if __name__ == "__main__":
    main()
