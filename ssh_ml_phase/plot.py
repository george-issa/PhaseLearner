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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
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
_CM_BETA    = "coolwarm"   # continuous parameter (β or α) colormap
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
    """
    base = results_dir or _cfg.RESULTS_DIR
    parts = [base, "figures", model or "unspecified", method]
    if tag:
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

def _betac_vline(ax, betac):
    """Vertical dotted line at FSS β_c plus a small label, no-op if NaN/None."""
    if betac is None or not np.isfinite(betac):
        return
    ax.axvline(betac, color="gray", ls=":", lw=1.0)
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
    if betac is not None and np.isfinite(betac):
        inset.axvline(betac, color="gray", ls=":", lw=0.8)
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
    """2D embedding scatter coloured by β with a diverging colormap centred
    at β_c.  Falls back to coolwarm when β_c is unavailable.
    Generic over PCA scores and UMAP coordinates via the xlabel/ylabel hooks.
    """
    if betac is not None and np.isfinite(betac):
        norm = TwoSlopeNorm(vcenter=float(betac),
                            vmin=float(betas.min()), vmax=float(betas.max()))
        cmap = "RdBu"
    else:
        norm = None
        cmap = "coolwarm"
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


def _diag_panel_pc1_loading(ax, loading, L, is_q, label):
    """PC1 loading — L×L imshow on the lattice or BZ when len(loading)=L²;
    otherwise a feature-index bar plot of |loading|."""
    if loading is None or len(loading) == 0:
        ax.set_visible(False)
        return
    if len(loading) == L * L:
        v = loading.reshape(L, L)
        if v.flat[np.argmax(np.abs(v))] < 0:
            v = -v
        amax = float(np.max(np.abs(v)))
        im = ax.imshow(v, origin="upper", cmap="RdBu_r",
                       vmin=-amax, vmax=amax, interpolation="nearest")
        plt.colorbar(im, ax=ax, pad=0.02).set_label(r"PC1 loading")
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
        ax.set_title(f"{label} --- PC1 loading on the "
                     + ("Brillouin zone" if is_q else "lattice"),
                     fontsize=10)
    else:
        v = loading
        if v.flat[np.argmax(np.abs(v))] < 0:
            v = -v
        ax.bar(np.arange(len(v)), v, color=_D_SCATTER, width=1.0)
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_xlabel(r"feature index $i$")
        ax.set_ylabel(r"PC1 loading $w_i$")
        ax.set_title(f"{label} --- PC1 loading vs feature index "
                     rf"($n_{{\mathrm{{feat}}}}={len(v)}$)",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")


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


def plot_pca_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir):
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
            b_s     = _safe(data, f"{key}_betas_sample")
            km_s    = _safe(data, f"{key}_km_labels_sample")
            var_P1  = _safe(data, f"{key}_var_P1")
            var_P2  = _safe(data, f"{key}_var_P2")
            ev      = _safe(data, f"{key}_explained_var")
            loading = _safe(data, f"{key}_pc1_loading")
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

            # ── embedding.pdf  (2×2 layout) ───────────────────────────────
            fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                                     constrained_layout=True)

            # Top row: embedding space, two views of the same data
            _diag_panel_p1p2_scatter(axes[0, 0], P1_s, P2_s, b_s, betac)
            if km_s is not None:
                _diag_panel_p1p2_clusters(axes[0, 1], P1_s, P2_s, km_s, b_s)
                # Lock cluster panel to the β-scatter limits so the two
                # views read as a before/after of the same point cloud.
                axes[0, 1].set_xlim(axes[0, 0].get_xlim())
                axes[0, 1].set_ylim(axes[0, 0].get_ylim())
            else:
                axes[0, 1].set_visible(False)

            # Bottom row: cluster fraction and its sharpness, vs β
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

            fig.suptitle(
                rf"{obs_label} --- PCA embedding  $|$  "
                rf"$\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=13)
            pdf_emb.savefig(fig, dpi=120)
            plt.close(fig)

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

    print(f"  Saved: {out_avg}")
    print(f"  Saved: {out_emb}")
    print(f"  Saved: {out_pc}")
    for _, p in legacy_pdfs:
        print(f"  Saved: {p}")
    return (out_avg, out_emb, out_pc)


# ── UMAP — beta sweep ─────────────────────────────────────────────────────────

def plot_umap_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir):
    """
    Multi-page PDF for UMAP beta-sweep: ``embedding.pdf`` — same 2×2
    diagnostic-style layout as the PCA embedding.
    Run tag is encoded by ``fig_dir`` (`<model>/umap/<tag>/`).
    """
    if not os.path.exists(npz_path):
        print(f"  [UMAP plot] Not found: {npz_path}"); return
    data      = np.load(npz_path, allow_pickle=False)
    obs_label = (model_info.get("observable") or "").replace("_", r"\_")
    out       = os.path.join(fig_dir, "embedding.pdf")

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
                xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$",
                title=r"UMAP --- coloured by $\beta$")
            if emb_km is not None:
                _diag_panel_p1p2_clusters(
                    axes[0, 1], emb_x, emb_y, emb_km, emb_b,
                    xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$",
                    title=r"UMAP --- coloured by cluster")
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
                rf"{obs_label} --- UMAP embedding  $|$  "
                rf"$\alpha={alpha:.4f}$  "
                rf"$\beta_c(\mathrm{{FSS}})={betac_str}$",
                fontsize=13)
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

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

def run_all_plots(tag, alphas, fss_dict, model_info,
                  alpha_sweep=False, betas=None,
                  skip_umap=False, results_dir=None):
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
        t_suf    = f"_{tag}" if tag else ""
        pca_npz  = os.path.join(npz_dir, f"pca_results{t_suf}.npz")
        umap_npz = os.path.join(npz_dir, f"umap_results{t_suf}.npz")

        print("\n[PCA beta-sweep]")
        plot_pca_beta_sweep(pca_npz, alphas, fss_dict, model_info, pca_dir)

        if not skip_umap:
            print("\n[UMAP beta-sweep]")
            plot_umap_beta_sweep(umap_npz, alphas, fss_dict, model_info, umap_dir)

        print("\n[Summary: β_c vs α]")
        plot_beta_c_summary(pca_npz, umap_npz, alphas, fss_dict, model_info,
                            summ_dir)

    print(f"\nFigures saved under:")
    print(f"  PCA     →  {pca_dir}")
    if not skip_umap:
        print(f"  UMAP    →  {umap_dir}")
    if not alpha_sweep:
        print(f"  Summary →  {summ_dir}")


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
        tag          = tag,
        alphas       = alphas,
        fss_dict     = fss_dict,
        model_info   = model_info,
        alpha_sweep  = alpha_sweep,
        betas        = betas,
        skip_umap    = skip_umap,
        results_dir  = results_dir,
    )


if __name__ == "__main__":
    main()
