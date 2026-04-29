"""
Plotting pipeline for ML phase detection results.

Reads .npz files produced by pca_method / umap_method and generates
multi-page PDFs, one page per alpha (beta-sweep) or per beta (alpha-sweep).

Beta-sweep PDFs  (saved to results/figures/plots_<tag>/):
  pca_susceptibility_<tag>.pdf  — Var(P1), Var(P2), and their derivatives
  pca_kmeans_<tag>.pdf          — PCA scatter + KMeans f_KVBS + derivative
  umap_<tag>.pdf                — UMAP embedding + f_KVBS + derivative
  beta_c_summary_<tag>.pdf      — β_c vs α from all methods

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

plt.rcParams.update({
    "font.size":         12,
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "mathtext.fontset":  "cm",
})

from . import config as _cfg
from .data import RESULTS_DIR, make_tag, result_filename


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


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _sigmoid(x, xc, w):
    return 1.0 / (1.0 + np.exp(-(x - xc) / w))


def _figures_dir(tag, results_dir=None):
    name = f"plots_{tag}" if tag else "plots"
    path = os.path.join(results_dir or RESULTS_DIR, "figures", name)
    os.makedirs(path, exist_ok=True)
    return path


def _info_str(mi):
    """Compact one-line parameter string for suptitles."""
    pairs = [("model", mi.get("model")), ("obs", mi.get("observable")),
             ("L",     mi.get("L")),     ("β_min", mi.get("beta_min")),
             ("β_max", mi.get("beta_max"))]
    return "  |  ".join(f"{k}={v}" for k, v in pairs if v is not None)


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
    ax.legend(loc="upper left")
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
    ax.legend()
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
    ax.plot(u_x, f_ord, "ko", ms=5, label=ylabel)
    if popt is not None:
        x_fine = np.linspace(u_x[0], u_x[-1], 300)
        ax.plot(x_fine, _sigmoid(x_fine, popt[0], popt[1]),
                "r-", lw=2,
                label=rf"sigmoid fit  $\beta_c={popt[0]:.2f}\pm{popt[1]:.2f}$")
    _vlines(ax, bc_arr, fss,
            idx_sm=idx_sm, idx_sig=idx_sig, idx_deriv=idx_deriv,
            param_label=xlabel)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
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
    ax.legend()
    ax.grid(True, alpha=0.3)


# ── PCA — beta sweep ──────────────────────────────────────────────────────────

def plot_pca_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir):
    """
    Two multi-page PDFs for PCA beta-sweep results:
      pca_susceptibility_<tag>.pdf  (Var panels)
      pca_kmeans_<tag>.pdf          (scatter + f_KVBS panels)
    """
    if not os.path.exists(npz_path):
        print(f"  [PCA plot] Not found: {npz_path}"); return
    data    = np.load(npz_path, allow_pickle=False)
    info    = _info_str(model_info)
    tag     = model_info.get("tag", "")
    t_suf   = f"_{tag}" if tag else ""

    out_var = os.path.join(fig_dir, f"pca_susceptibility{t_suf}.pdf")
    out_km  = os.path.join(fig_dir, f"pca_kmeans{t_suf}.pdf")

    with PdfPages(out_var) as pdf_var, PdfPages(out_km) as pdf_km:
        for alpha in alphas:
            key    = f"a{alpha:.4f}"
            fss    = float(fss_dict.get(alpha, np.nan))
            bc_arr = _safe(data, f"{key}_beta_c")

            u_beta  = _safe(data, f"{key}_u_beta")
            if u_beta is None:
                continue

            # ── Susceptibility page ───────────────────────────────────────
            var_P1   = data[f"{key}_var_P1"]
            var_P1sm = data[f"{key}_var_P1_sm"]
            var_P2   = data[f"{key}_var_P2"]
            var_P2sm = data[f"{key}_var_P2_sm"]
            d_beta1  = _safe(data, f"{key}_var1_deriv_beta")
            d_vals1  = _safe(data, f"{key}_var1_deriv_vals")
            d_beta2  = _safe(data, f"{key}_var2_deriv_beta")
            d_vals2  = _safe(data, f"{key}_var2_deriv_vals")

            fss_str  = f"{fss:.3f}" if not np.isnan(fss) else "N/A"
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            fig.suptitle(
                f"PCA susceptibility  |  α={alpha:.4f}  β_c(FSS)={fss_str}\n{info}",
                fontsize=10)

            _panel_var(axes[0], u_beta, var_P1, var_P1sm, var_P2, var_P2sm,
                       bc_arr=bc_arr, fss=fss if not np.isnan(fss) else None)
            axes[0].set_title(r"$\mathrm{Var}(P_1)$ and $\mathrm{Var}(P_2)$ vs $\beta$")

            # Var(P1) alone with estimator lines
            axes[1].plot(u_beta, var_P1,   "o", ms=4, color=_C_VAR1, alpha=0.5,
                         label=r"$\mathrm{Var}(P_1)$ raw")
            axes[1].plot(u_beta, var_P1sm, "-", lw=2, color=_C_VAR1,
                         label="smoothed")
            _vlines(axes[1], bc_arr, fss if not np.isnan(fss) else None)
            axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel(r"$\mathrm{Var}(P_1)$")
            axes[1].set_title(r"$\mathrm{Var}(P_1)$ — estimators")
            axes[1].legend(); axes[1].grid(True, alpha=0.3)

            # Derivatives
            if d_beta1 is not None and d_vals2 is not None:
                _panel_deriv(axes[2], d_beta1, d_vals1, d_vals2)
            else:
                axes[2].set_visible(False)
            axes[2].set_title(r"$d\,\mathrm{Var}/d\beta$  (zero-crossing $\approx\beta_c$)")

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf_var.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── KMeans page ───────────────────────────────────────────────
            P1_s  = _safe(data, f"{key}_P1_sample")
            P2_s  = _safe(data, f"{key}_P2_sample")
            b_s   = _safe(data, f"{key}_betas_sample")
            km_s  = _safe(data, f"{key}_km_labels_sample")
            f_kv  = _safe(data, f"{key}_f_kvbs_km")
            popt  = _safe(data, f"{key}_km_sig_popt")
            d_kb  = _safe(data, f"{key}_km_deriv_beta")
            d_kv  = _safe(data, f"{key}_km_deriv_vals")

            if P1_s is None or f_kv is None:
                continue

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(
                f"PCA KMeans  |  α={alpha:.4f}  β_c(FSS)={fss_str}\n{info}",
                fontsize=10)

            _panel_scatter(axes[0], P1_s, P2_s, b_s, _CM_BETA, r"$\beta$",
                           title=r"$P_1$ vs $P_2$ — coloured by $\beta$",
                           xlabel=r"$P_1$", ylabel=r"$P_2$")

            _panel_scatter(axes[1], P1_s, P2_s, km_s.astype(float), _CM_CLUSTER,
                           "cluster", title=r"$P_1$ vs $P_2$ — $k$-means clusters",
                           xlabel=r"$P_1$", ylabel=r"$P_2$")

            _panel_f_ordered(axes[2], u_beta, f_kv, popt, bc_arr,
                             fss if not np.isnan(fss) else None,
                             ylabel=r"$f_\mathrm{KVBS}$", idx_sig=2, idx_deriv=4)
            axes[2].set_title(r"$f_\mathrm{KVBS}$ vs $\beta$  (KMeans sigmoid)")

            if d_kb is not None and d_kv is not None:
                _panel_f_deriv(axes[3], d_kb, d_kv, popt)
            else:
                axes[3].set_visible(False)
            axes[3].set_title(r"$df_\mathrm{KVBS}/d\beta$")

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf_km.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: {out_var}")
    print(f"  Saved: {out_km}")
    return out_var, out_km


# ── UMAP — beta sweep ─────────────────────────────────────────────────────────

def plot_umap_beta_sweep(npz_path, alphas, fss_dict, model_info, fig_dir):
    """
    Multi-page PDF for UMAP beta-sweep:
      umap_<tag>.pdf  (4 panels per alpha)
    """
    if not os.path.exists(npz_path):
        print(f"  [UMAP plot] Not found: {npz_path}"); return
    data   = np.load(npz_path, allow_pickle=False)
    info   = _info_str(model_info)
    tag    = model_info.get("tag", "")
    t_suf  = f"_{tag}" if tag else ""
    out    = os.path.join(fig_dir, f"umap{t_suf}.pdf")

    with PdfPages(out) as pdf:
        for alpha in alphas:
            key    = f"a{alpha:.4f}"
            fss    = float(fss_dict.get(alpha, np.nan))
            fss_str = f"{fss:.3f}" if not np.isnan(fss) else "N/A"
            bc_arr = _safe(data, f"{key}_beta_c")
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

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(
                f"UMAP  |  α={alpha:.4f}  β_c(FSS)={fss_str}\n{info}",
                fontsize=10)

            _panel_scatter(axes[0], emb_x, emb_y, emb_b, _CM_BETA, r"$\beta$",
                           title=r"UMAP — coloured by $\beta$",
                           xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$")

            _panel_scatter(axes[1], emb_x, emb_y,
                           emb_km.astype(float), _CM_CLUSTER, "cluster",
                           title=r"UMAP — $k$-means clusters",
                           xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$")

            # UMAP beta_c layout: [km_sig(0), km_sig_err(1), km_deriv(2), best(3)]
            _panel_f_ordered(axes[2], u_beta, f_kv, popt, bc_arr,
                             fss if not np.isnan(fss) else None,
                             ylabel=r"$f_\mathrm{KVBS}$", idx_sm=None, idx_sig=0, idx_deriv=2)
            axes[2].set_title(r"$f_\mathrm{KVBS}$ vs $\beta$  (UMAP $k$-means)")

            if d_kb is not None and d_kv is not None:
                _panel_f_deriv(axes[3], d_kb, d_kv, popt)
            else:
                axes[3].set_visible(False)
            axes[3].set_title(r"$df_\mathrm{KVBS}/d\beta$")

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: {out}")
    return out


# ── Summary: β_c vs α ────────────────────────────────────────────────────────

def plot_beta_c_summary(pca_npz, umap_npz, alphas, fss_dict, model_info, fig_dir):
    """
    Single figure: β_c vs α, overlaying PCA, UMAP, and FSS estimates.
    """
    tag   = model_info.get("tag", "")
    t_suf = f"_{tag}" if tag else ""
    out   = os.path.join(fig_dir, f"beta_c_summary{t_suf}.pdf")

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
    ax.set_title(f"β_c vs α — all methods\n{_info_str(model_info)}", fontsize=10)

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
    tag    = model_info.get("tag", "")
    t_suf  = f"_{tag}" if tag else ""
    out    = os.path.join(fig_dir, f"pca_alpha_sweep{t_suf}.pdf")

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
                ac_str = f"  |  α_c(KM sig)={ac_arr[2]:.3f}"

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(
                f"PCA α-sweep  |  β={beta:.2f}{ac_str}\n{info}",
                fontsize=10)

            if P1_s is not None:
                _panel_scatter(axes[0], P1_s, P2_s, a_s, _CM_BETA, r"$\alpha$",
                               title=r"$P_1$ vs $P_2$ — coloured by $\alpha$",
                               xlabel=r"$P_1$", ylabel=r"$P_2$")

                _panel_scatter(axes[1], P1_s, P2_s,
                               km_s.astype(float), _CM_CLUSTER, "cluster",
                               title=r"$P_1$ vs $P_2$ — $k$-means clusters",
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
    tag    = model_info.get("tag", "")
    t_suf  = f"_{tag}" if tag else ""
    out    = os.path.join(fig_dir, f"umap_alpha_sweep{t_suf}.pdf")

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
                ac_str = f"  |  α_c(KM sig)={ac_arr[0]:.3f}"

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(
                f"UMAP α-sweep  |  β={beta:.2f}{ac_str}\n{info}",
                fontsize=10)

            if emb_x is not None:
                _panel_scatter(axes[0], emb_x, emb_y, emb_a, _CM_BETA, r"$\alpha$",
                               title=r"UMAP — coloured by $\alpha$",
                               xlabel=r"UMAP$_1$", ylabel=r"UMAP$_2$")

                _panel_scatter(axes[1], emb_x, emb_y,
                               emb_km.astype(float), _CM_CLUSTER, "cluster",
                               title=r"UMAP — $k$-means clusters",
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
    rd      = results_dir or RESULTS_DIR
    fig_dir = _figures_dir(tag, rd)
    print(f"\n{'='*60}")
    print(f"Plotting  →  {fig_dir}")
    print(f"{'='*60}")

    if alpha_sweep:
        if not betas:
            print("  [plot] alpha_sweep=True but no betas given — skipping."); return

        pca_npz  = os.path.join(rd, "pca_results_alpha_sweep.npz")
        umap_npz = os.path.join(rd, "umap_results_alpha_sweep.npz")
        if tag:
            # also check tag-specific file
            t_pca  = os.path.join(rd, f"pca_results_{tag}.npz")
            t_umap = os.path.join(rd, f"umap_results_{tag}.npz")
            if os.path.exists(t_pca):  pca_npz  = t_pca
            if os.path.exists(t_umap): umap_npz = t_umap

        print("\n[PCA alpha-sweep]")
        plot_pca_alpha_sweep(pca_npz, betas, model_info, fig_dir)

        if not skip_umap:
            print("\n[UMAP alpha-sweep]")
            plot_umap_alpha_sweep(umap_npz, betas, model_info, fig_dir)

    else:
        t_suf    = f"_{tag}" if tag else ""
        pca_npz  = os.path.join(rd, f"pca_results{t_suf}.npz")
        umap_npz = os.path.join(rd, f"umap_results{t_suf}.npz")

        print("\n[PCA beta-sweep]")
        plot_pca_beta_sweep(pca_npz, alphas, fss_dict, model_info, fig_dir)

        if not skip_umap:
            print("\n[UMAP beta-sweep]")
            plot_umap_beta_sweep(umap_npz, alphas, fss_dict, model_info, fig_dir)

        print("\n[Summary: β_c vs α]")
        plot_beta_c_summary(pca_npz, umap_npz, alphas, fss_dict, model_info, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


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
