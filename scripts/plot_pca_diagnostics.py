"""
PCA diagnostics for the static (equal-time) correlators (attractive Hubbard pilot).

Two diagnostics, one PDF page each:

  1. P1, P2 vs β  — scatter of all DQMC samples plus per-β mean ± std.
     Reveals the order-parameter shape of PC1 (exponential ramp at high β)
     and whether PC2 carries any β-dependent signal.

  2. PCA eigenvalue spectrum — explained-variance ratio per component on
     a log scale.  A standalone PC1 (one big eigenvalue, large gap to PC2)
     means the data really lives on a 1-D order-parameter axis.

Reads the existing results/npz/pca_results_{density,pair}_{position,momentum}.npz —
PCA stats (scores, eigenvalues) are independent of the downstream clustering,
so no recompute is required.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde, skew, kurtosis

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
NPZ_DIR      = os.path.join(RESULTS_DIR, "npz")
# Per-observable PDFs land alongside the canonical pipeline plots:
#   results/figures/<model>/pca/<observable>/diagnostics.pdf
MODEL_NAME   = "attractive_hubbard"


def _run_dir(obs):
    """results/figures/<model>/pca/<observable>/  — same folder the pipeline
    PDFs (pca_susceptibility.pdf, pca_kmeans.pdf) write into."""
    d = os.path.join(RESULTS_DIR, "figures", MODEL_NAME, "pca", obs)
    os.makedirs(d, exist_ok=True)
    return d

OBSERVABLES = [
    ("density_position",  r"$S_{n}(\mathbf{r})$ (real space)"),
    ("density_momentum",  r"$S_{n}(\mathbf{q})$ ($\mathbf{q}$-space)"),
    ("pair_position",     r"$P_{\mathrm{SC}}(\mathbf{r})$ (real space)"),
    ("pair_momentum",     r"$P_{\mathrm{SC}}(\mathbf{q})$ ($\mathbf{q}$-space)"),
]
ALPHA_KEY = "a4.0000"
BETA_C    = 8.5


def load(obs):
    fn = os.path.join(NPZ_DIR, f"pca_results_{obs}.npz")
    d  = np.load(fn, allow_pickle=True)
    return {
        "P1_sample":     d[f"{ALPHA_KEY}_P1_sample"],
        "P2_sample":     d[f"{ALPHA_KEY}_P2_sample"],
        "betas_sample":  d[f"{ALPHA_KEY}_betas_sample"],
        "u_beta":        d[f"{ALPHA_KEY}_u_beta"],
        "var_P1":        d[f"{ALPHA_KEY}_var_P1"],     # per-β var, full data
        "var_P2":        d[f"{ALPHA_KEY}_var_P2"],
        "explained_var": d[f"{ALPHA_KEY}_explained_var"],
    }


def panel_pc_vs_beta(ax, betas, P_samples, u_beta, ylabel, title):
    """Scatter of (β, PC) for all samples plus per-β mean ± std band.

    The inset (per-β mean alone) and the legend are placed in the two
    lowest-density quadrants of the *already-auto-scaled* axes — y-limits
    are not modified.  This way wide std bands at intermediate β still
    determine the y-extent, and the inset/legend just dodge wherever the
    scatter is sparsest.
    """
    ax.scatter(betas, P_samples, s=2, alpha=0.15, color="#1f77b4",
               rasterized=True, label=r"per-sample")
    means = np.array([P_samples[betas == b].mean() for b in u_beta])
    stds  = np.array([P_samples[betas == b].std()  for b in u_beta])
    ax.plot(u_beta, means, "o-", color="#d62728", lw=1.5, ms=4,
            label=r"$\langle\cdot\rangle_\beta\ \pm\ \mathrm{std}$")
    ax.fill_between(u_beta, means - stds, means + stds,
                    alpha=0.25, color="#d62728")
    ax.axvline(BETA_C, color="gray", ls=":", lw=1.0)
    ax.text(BETA_C, 0.98, rf"\ $\beta_c\!\approx\!{BETA_C}$",
            color="gray", fontsize=8, va="top", ha="left",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)

    # ── Pick inset + legend corners from the existing axis limits ────────
    # We do NOT call ax.relim()/autoscale_view(); the auto-scale that
    # matplotlib applied during scatter()/plot()/fill_between() is what we
    # want to keep.
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

    # ── Inset: per-β mean only (no scatter) ──────────────────────────────
    inset = ax.inset_axes([ix, iy, inset_w, inset_h])
    inset.plot(u_beta, means, "o-", color="#d62728", lw=1.4, ms=3)
    inset.axvline(BETA_C, color="gray", ls=":", lw=0.8)
    inset.set_title(r"$\langle\cdot\rangle_\beta$ only",
                    fontsize=7, pad=2)
    inset.tick_params(labelsize=6, length=2)
    inset.grid(alpha=0.3)
    inset.set_xlabel(r"$\beta$", fontsize=7, labelpad=1)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.92)


def panel_pc_overlay(ax, P1_samples, P2_samples, betas, u_beta, title):
    """Overlay per-β means of P1 and P2 on the SAME axes, both z-scored by
    their own global std.  σ(P_k)² = λ_k (PCA eigenvalue), so dividing each
    PC by its global std puts them in dimensionless 'number of σ' units —
    the only fair shared scale for PC1 and PC2 given their ~16× variance
    gap.  Crossings on this plot mark β values where the two PCs carry
    equal (signed) signal in std-units."""
    s1 = P1_samples.std()
    s2 = P2_samples.std()
    m1 = np.array([P1_samples[betas == b].mean() for b in u_beta]) / s1
    m2 = np.array([P2_samples[betas == b].mean() for b in u_beta]) / s2

    ax.plot(u_beta, m1, "o-", color="#d62728", lw=1.5, ms=4,
            label=r"$\langle P_1\rangle_\beta / \sigma_1$")
    ax.plot(u_beta, m2, "s-", color="#2ca02c", lw=1.5, ms=4,
            label=r"$\langle P_2\rangle_\beta / \sigma_2$")

    # Mark crossings: linear-interpolate between adjacent betas where (m1-m2)
    # changes sign.  Yellow dots stay on the curves; the β_× values are
    # collected in a single corner box so they don't pile on top of the
    # β_c axis label or each other.
    diff = m1 - m2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    crossings = []
    for i in sign_changes:
        b0, b1 = u_beta[i], u_beta[i+1]
        d0, d1 = diff[i], diff[i+1]
        b_cross = b0 - d0 * (b1 - b0) / (d1 - d0)
        y_cross = m1[i] + (m1[i+1] - m1[i]) * (b_cross - b0) / (b1 - b0)
        ax.plot(b_cross, y_cross, "o", ms=9,
                mfc="yellow", mec="black", zorder=5)
        crossings.append(b_cross)

    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.axvline(BETA_C, color="gray", ls=":", lw=1.0)
    ax.text(BETA_C, 0.98, rf"\ $\beta_c\!\approx\!{BETA_C}$",
            color="gray", fontsize=8, va="top", ha="left",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle P_k\rangle_\beta / \sigma_k$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Place the crossing values in the upper-left corner — far from the
    # β_c label (top, anchored at BETA_C in the middle of the x-axis) and
    # far from the legend (lower-right).
    if crossings:
        info = "\n".join(rf"$\beta_\times={b:.2f}$" for b in crossings)
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9))


def panel_pc_var_overlay(ax, var_P1, var_P2, P1_samples, P2_samples,
                         u_beta, title):
    """Overlay the per-β variances of P1 and P2, each normalized by its own
    global eigenvalue λ_k = σ_k².  The result is dimensionless and the two
    PCs share a fair scale.  Above 1 = locally more spread than the global
    cloud (susceptibility peak); below 1 = locally tighter than global.
    The argmax of var(P_k|β)/λ_k is the PCA-susceptibility estimator of β_c.
    """
    lambda1 = P1_samples.var()    # ≈ global eigenvalue λ_1
    lambda2 = P2_samples.var()    # ≈ global eigenvalue λ_2
    v1 = var_P1 / lambda1
    v2 = var_P2 / lambda2

    ax.plot(u_beta, v1, "o-", color="#d62728", lw=1.5, ms=4,
            label=r"$\mathrm{Var}(P_1|\beta)/\lambda_1$")
    ax.plot(u_beta, v2, "s-", color="#2ca02c", lw=1.5, ms=4,
            label=r"$\mathrm{Var}(P_2|\beta)/\lambda_2$")
    ax.axhline(1.0, color="black", lw=0.7, alpha=0.5, ls="--",
               label=r"global $= 1$")
    ax.axvline(BETA_C, color="gray", ls=":", lw=1.0)
    ax.text(BETA_C, 0.98, rf"\ $\beta_c\!\approx\!{BETA_C}$",
            color="gray", fontsize=8, va="top", ha="left",
            transform=ax.get_xaxis_transform())

    # Mark argmax of each — PCA-susceptibility estimator of β_c.
    # Yellow markers stay on the curves; the β_c^(k) values go into two
    # small colour-matched boxes in the upper-left corner so they don't
    # overlap the β_c axis label (top, attached to the BETA_C vertical)
    # or stack on top of each other.
    bc1 = u_beta[np.argmax(v1)]
    bc2 = u_beta[np.argmax(v2)]
    ax.plot(bc1, v1.max(), "o", ms=10, mfc="yellow", mec="#d62728",
            zorder=5)
    ax.plot(bc2, v2.max(), "s", ms=10, mfc="yellow", mec="#2ca02c",
            zorder=5)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\mathrm{Var}(P_k|\beta)/\lambda_k$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax.text(0.02, 0.98, rf"$\beta_c^{{(1)}}={bc1:.1f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#d62728",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#d62728", alpha=0.9))
    ax.text(0.02, 0.88, rf"$\beta_c^{{(2)}}={bc2:.1f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#2ca02c", alpha=0.9))


def panel_skew_kurt(ax, P_samples, betas, u_beta, title, pc_label=r"P_1"):
    """Per-β skewness and excess kurtosis of a PC, on twin y-axes.

    Reference levels (Gaussian): skewness = 0, excess kurtosis = 0.
    Skewness > 0 means right-leaning (long right tail); excess kurt > 0
    means heavier tails than Gaussian.  Diagnostics for whether the
    per-β distribution drifts away from the KMeans/GMM Gaussian assumption
    as β grows.
    """
    skew_vals = np.array([skew(P_samples[betas == b]) for b in u_beta])
    kurt_vals = np.array([kurtosis(P_samples[betas == b], fisher=True)
                          for b in u_beta])

    color_skew = "#9467bd"   # purple
    color_kurt = "#ff7f0e"   # orange

    l1, = ax.plot(u_beta, skew_vals, "o-", color=color_skew, lw=1.5, ms=4,
                  label=rf"skewness$({pc_label})$")
    ax.set_ylabel("skewness", color=color_skew)
    ax.tick_params(axis="y", labelcolor=color_skew)
    ax.axhline(0, color=color_skew, ls=":", lw=0.6, alpha=0.5)

    ax2 = ax.twinx()
    l2, = ax2.plot(u_beta, kurt_vals, "s-", color=color_kurt, lw=1.5, ms=4,
                   label=rf"excess kurt$({pc_label})$")
    ax2.set_ylabel("excess kurtosis", color=color_kurt)
    ax2.tick_params(axis="y", labelcolor=color_kurt)
    ax2.axhline(0, color=color_kurt, ls=":", lw=0.6, alpha=0.5)

    ax.axvline(BETA_C, color="gray", ls=":", lw=1.0)
    ax.text(BETA_C, 0.98, rf"\ $\beta_c\!\approx\!{BETA_C}$",
            color="gray", fontsize=8, va="top", ha="left",
            transform=ax.get_xaxis_transform())

    ax.set_xlabel(r"$\beta$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2], loc="best", fontsize=8)


def panel_spectrum(ax, ev, title):
    """Explained-variance ratio per PC index on a semilog axis."""
    k = np.arange(1, len(ev) + 1)
    ax.plot(k, ev, "o-", lw=1.4, ms=5, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("PC index $k$")
    ax.set_ylabel(r"$\lambda_k / \sum_j \lambda_j$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    # Annotate the dominant eigenvalue and the gap to the next.
    ax.annotate(rf"$\lambda_1/\sum = {ev[0]:.3f}$",
                xy=(1, ev[0]), xytext=(1.6, ev[0] * 0.4),
                fontsize=9, color="#1f77b4",
                arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.6))
    if len(ev) >= 2:
        gap = ev[0] / ev[1]
        ax.annotate(rf"$\lambda_1/\lambda_2 \approx {gap:.0f}$",
                    xy=(2, ev[1]), xytext=(3.5, ev[1] * 4),
                    fontsize=9, color="#2ca02c",
                    arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.6))


def get_pc1_loading(obs):
    """Re-fit PCA on the full data to extract PC1's loading vector (length L²).
    PCA stats are deterministic, so this matches whatever the saved .npz used."""
    from ssh_ml_phase import config as _cfg
    from ssh_ml_phase.data import load_beta_sweep
    from sklearn.decomposition import PCA as _PCA
    _cfg.apply_model("attractive_hubbard")
    pat  = _cfg.OBSERVABLE_DATA_PATTERN
    ddir = os.path.join(_cfg.DATA_DIR, pat["data_subdir"].format(obs=obs))
    ft   = pat["file_template"].format(obs=obs)
    X, _ = load_beta_sweep(4.0, data_dir=ddir, file_template=ft)
    Xc   = X - X.mean(axis=0)
    p    = _PCA(n_components=2).fit(Xc)
    return p.components_[0]   # shape (L²,)


def page_p1p2_scatter(pdf, obs, label):
    """Single-observable (P1, P2) scatter coloured by β (RdBu, β_c-centered)."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.6))
    d = load(obs)
    norm = TwoSlopeNorm(vcenter=BETA_C,
                        vmin=d["u_beta"].min(),
                        vmax=d["u_beta"].max())
    sc = ax.scatter(d["P1_sample"], d["P2_sample"], c=d["betas_sample"],
                    cmap="RdBu", norm=norm,
                    s=4, alpha=0.7, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\beta$ \quad (red = hot, blue = cold)", fontsize=9)
    cb.ax.axhline(BETA_C, color="black", lw=0.8)
    cb.ax.text(0.5, BETA_C, rf"\ $\beta_c$", color="black", fontsize=8,
               ha="left", va="center",
               transform=cb.ax.get_yaxis_transform())
    ax.set_xlabel(r"$P_1$")
    ax.set_ylabel(r"$P_2$")
    ax.set_title(f"{label} --- $(P_1, P_2)$ coloured by $\\beta$",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle(r"PCA scatter --- diverging colormap centred at $\beta_c$",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def page_pc1_loading(pdf, obs, label, L=8):
    """Single-observable PC1 loading reshaped on lattice (real) or BZ (q-space)."""
    v = get_pc1_loading(obs).reshape(L, L)
    if v.flat[np.argmax(np.abs(v))] < 0:
        v = -v
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.6))
    is_q = "momentum" in obs
    amax = np.max(np.abs(v))
    im = ax.imshow(v, origin="upper", cmap="RdBu_r",
                   vmin=-amax, vmax=amax, interpolation="nearest")
    plt.colorbar(im, ax=ax, pad=0.02).set_label(r"PC1 loading")
    ax.set_title(f"{label} --- PC1 loading on the "
                 + ("Brillouin zone" if is_q else "lattice"), fontsize=10)
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
    fig.suptitle(r"Physical interpretation of PC1: which sites/$\mathbf{q}$ "
                 r"contribute to the order parameter", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def page_p1_ridgeline(pdf, obs, label):
    """Single-observable stacked per-β KDE of P1 (ridgeline plot)."""
    d = load(obs)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 8))
    u_beta = d["u_beta"]
    cmap   = plt.get_cmap("RdBu")
    norm   = TwoSlopeNorm(vcenter=BETA_C,
                          vmin=u_beta.min(), vmax=u_beta.max())

    lo = np.percentile(d["P1_sample"], 1)
    hi = np.percentile(d["P1_sample"], 99)
    pad = 0.05 * (hi - lo)
    x = np.linspace(lo - pad, hi + pad, 400)

    for i, b in enumerate(u_beta):
        samples = d["P1_sample"][d["betas_sample"] == b]
        if len(samples) < 5:
            continue
        try:
            kde  = gaussian_kde(samples)
            dens = kde(x)
        except np.linalg.LinAlgError:
            continue
        dens  = dens / dens.max() * 0.85
        color = cmap(norm(b))
        ax.fill_between(x, i, i + dens, alpha=0.6, color=color, lw=0)
        ax.plot(x, i + dens, color=color, lw=0.6)

    bc_idx = float(np.argmin(np.abs(u_beta - BETA_C)))
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
    fig.suptitle(r"Per-$\beta$ shape of $P_1$ --- Gaussian? bimodal? "
                 r"heavy-tailed?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def render_observable_pdf(obs, label):
    """Build the per-observable diagnostics PDF and save it under
    ``results/figures/<model>/pca/<obs>/diagnostics.pdf`` — alongside the
    canonical pipeline plots for that run."""
    d        = load(obs)
    out_dir  = _run_dir(obs)
    out      = os.path.join(out_dir, "diagnostics.pdf")
    with PdfPages(out) as pdf:
        # ── Page 1: 4×1 column of scalar diagnostics for this observable ────
        fig, axes = plt.subplots(4, 1, figsize=(7.5, 16))
        panel_pc_vs_beta(axes[0], d["betas_sample"], d["P1_sample"],
                         d["u_beta"], r"$P_1$",
                         f"{label} --- $P_1$ vs $\\beta$")
        panel_pc_vs_beta(axes[1], d["betas_sample"], d["P2_sample"],
                         d["u_beta"], r"$P_2$",
                         f"{label} --- $P_2$ vs $\\beta$")
        panel_pc_overlay(axes[2], d["P1_sample"], d["P2_sample"],
                         d["betas_sample"], d["u_beta"],
                         f"{label} --- means of $P_1, P_2$ in std units")
        panel_pc_var_overlay(axes[3], d["var_P1"], d["var_P2"],
                             d["P1_sample"], d["P2_sample"], d["u_beta"],
                             f"{label} --- variances of $P_1, P_2$ "
                             r"(susceptibility, $/\lambda_k$)")
        fig.suptitle(rf"PCA diagnostics --- {obs}  "
                     r"(attractive Hubbard, $U=-4$, $\langle n\rangle=0.875$, $L=8$)",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

        # ── Pages 2–4: scatter, PC1 loading, ridgeline ──────────────────────
        page_p1p2_scatter(pdf, obs, label)
        page_pc1_loading(pdf, obs, label)
        page_p1_ridgeline(pdf, obs, label)

        # ── Page 5: PCA eigenvalue spectrum on its own page ─────────────────
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        panel_spectrum(ax, d["explained_var"], f"{label} --- PCA spectrum")
        fig.tight_layout()
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

    print(f"wrote {out}")


def main():
    for obs, label in OBSERVABLES:
        render_observable_pdf(obs, label)

    # ── Numerical summaries to stdout ──────────────────────────────────────
    print("\nEigenvalue summary:")
    for obs, _ in OBSERVABLES:
        d = load(obs)
        ev = d["explained_var"]
        print(f"  {obs:>30s}  λ1={ev[0]:.4f}  λ2={ev[1]:.4f}  "
              f"λ1/λ2={ev[0]/ev[1]:.1f}  cum(top-3)={ev[:3].sum():.4f}")

    # Skewness / excess kurtosis of P1 at five sample β values per observable.
    # The full panel was removed from page 1; this stdout block keeps the
    # numbers visible without re-running anything else.
    print("\nP1 higher-order moments (skewness, excess kurtosis):")
    for obs, _ in OBSERVABLES:
        d  = load(obs)
        ub = d["u_beta"]
        sample_betas = [ub.min(), ub[len(ub)//4], ub[len(ub)//2],
                        ub[3*len(ub)//4], ub.max()]
        print(f"  {obs}")
        print(f"    {'β':>5s}  {'skew(P1)':>10s}  {'ex kurt(P1)':>12s}")
        for b in sample_betas:
            s = d["P1_sample"][d["betas_sample"] == b]
            print(f"    {b:>5.1f}  {skew(s):>+10.3f}  {kurtosis(s, fisher=True):>+12.3f}")


if __name__ == "__main__":
    main()
