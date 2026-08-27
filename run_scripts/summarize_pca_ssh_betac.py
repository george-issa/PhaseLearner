#!/usr/bin/env python3
"""
Summarize PCA β_c across all 32 oSSH observables and rank them by agreement
with the FSS reference.

Reads:  results/npz/pca_results_<obs>_sid5.npz   (run_pca_ssh_all_obs.py output)
Writes: results/ssh_pca_betac_summary.csv        (full table)
        results/figures/ssh/pca_betac_summary.pdf/.png   (heatmaps)

Two estimators are tabulated (β_c array layout in pca_method.run_all):
  best   = smoothed argmax Var(P1)   (index 5)   — canonical susceptibility peak
  km_sig = KMeans sigmoid fit        (index 2)

Observables are ranked by RMS relative error of `best` vs FSS over the alphas
that have an FSS reference (0.240, 0.245, 0.250, 0.255; 0.2425 has none).

Run:  python run_scripts/summarize_pca_ssh_betac.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase import config as cfg

cfg.apply_model("ssh")

ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]
AKEYS = [f"a{a:.4f}" for a in ALPHAS]
FSS = np.array([cfg.FSS_DICT.get(a, np.nan) for a in ALPHAS])
FSS_MASK = ~np.isnan(FSS)

BEST, KM = 5, 2   # indices into the saved _beta_c array
NPZ_DIR = cfg.NPZ_DIR
OUT_CSV = os.path.join(cfg.RESULTS_DIR, "ssh_pca_betac_summary.csv")
FIG_DIR = os.path.join(cfg.RESULTS_DIR, "figures", "ssh")


def load_betac():
    """Return obs -> (best[5], kmsig[5]) arrays over ALPHAS (NaN if missing)."""
    out = {}
    for obs in cfg.OBSERVABLES:
        path = os.path.join(NPZ_DIR, f"pca_results_{obs}_sid5.npz")
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        d = np.load(path, allow_pickle=False)
        best = np.array([d[k + "_beta_c"][BEST] if k + "_beta_c" in d.files
                         else np.nan for k in AKEYS])
        kms = np.array([d[k + "_beta_c"][KM] if k + "_beta_c" in d.files
                        else np.nan for k in AKEYS])
        out[obs] = (best, kms)
    return out


def rmse_vs_fss(vals):
    """RMS relative error of vals vs FSS over the FSS-defined alphas."""
    rel = (vals[FSS_MASK] - FSS[FSS_MASK]) / FSS[FSS_MASK]
    return float(np.sqrt(np.nanmean(rel ** 2)))


def main():
    data = load_betac()
    rows = []
    for obs, (best, kms) in data.items():
        rows.append((obs, best, kms, rmse_vs_fss(best), rmse_vs_fss(kms)))
    # rank by RMSE of the canonical `best` estimator (lower = better agreement)
    rows.sort(key=lambda r: r[3])

    # ── console table ────────────────────────────────────────────────────────
    hdr = (f"{'rank':<5}{'observable':<38}{'estimator':<8}"
           + "".join(f"a{a:<7.4f}" for a in ALPHAS) + f"{'RMSE':>8}")
    print(hdr)
    print(f"{'':<51}" + "".join(f"{v:<8.1f}" if m else f"{'--':<8}"
                                 for v, m in zip(FSS, FSS_MASK))
          + "   (FSS reference)")
    print("-" * len(hdr))
    for rank, (obs, best, kms, rb, rk) in enumerate(rows, 1):
        print(f"{rank:<5}{obs:<38}{'best':<8}"
              + "".join(f"{v:<8.1f}" for v in best) + f"{rb:>8.2f}")
        print(f"{'':<5}{'':<38}{'km_sig':<8}"
              + "".join(f"{v:<8.1f}" for v in kms) + f"{rk:>8.2f}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        cols = (["rank", "observable", "estimator"]
                + [f"betac_a{a:.4f}" for a in ALPHAS] + ["rmse_vs_fss"])
        f.write(",".join(cols) + "\n")
        f.write(",".join(["", "FSS_reference", ""]
                         + [f"{v:.3f}" if m else "" for v, m in zip(FSS, FSS_MASK)]
                         + [""]) + "\n")
        for rank, (obs, best, kms, rb, rk) in enumerate(rows, 1):
            f.write(",".join([str(rank), obs, "best"]
                             + [f"{v:.3f}" for v in best] + [f"{rb:.4f}"]) + "\n")
            f.write(",".join(["", obs, "km_sig"]
                             + [f"{v:.3f}" for v in kms] + [f"{rk:.4f}"]) + "\n")
    print(f"\nWrote {OUT_CSV}")

    # ── heatmaps ─────────────────────────────────────────────────────────────
    _heatmaps(rows)


def _heatmaps(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    obs_names = [r[0] for r in rows]
    best = np.array([r[1] for r in rows])            # (32, 5)
    n = len(obs_names)

    # relative error vs FSS (only FSS alphas), signed
    rel = np.full_like(best, np.nan)
    rel[:, FSS_MASK] = (best[:, FSS_MASK] - FSS[FSS_MASK]) / FSS[FSS_MASK]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 0.34 * n + 1.5),
        gridspec_kw={"width_ratios": [1.0, 0.8]})

    # (a) β_c values
    ax = axes[0]
    im = ax.imshow(best, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([f"{a:.4f}" for a in ALPHAS], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(obs_names, fontsize=7)
    ax.set_title(r"PCA $\beta_c$ (best = smoothed argmax Var$(P_1)$)")
    ax.set_xlabel(r"$\alpha$")
    for i in range(n):
        for j in range(len(ALPHAS)):
            ax.text(j, i, f"{best[i, j]:.0f}", ha="center", va="center",
                    color="w", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\beta_c$")

    # (b) signed relative error vs FSS
    ax = axes[1]
    vmax = np.nanmax(np.abs(rel)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(rel, aspect="auto", cmap="RdBu_r", norm=norm)
    fss_cols = [j for j, m in enumerate(FSS_MASK) if m]
    ax.set_xticks(range(len(fss_cols)))
    ax.set_xticklabels([f"{ALPHAS[j]:.4f}\n(FSS {FSS[j]:.1f})" for j in fss_cols],
                       rotation=0, fontsize=7)
    ax.set_xlim(min(fss_cols) - 0.5, max(fss_cols) + 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    ax.set_title(r"signed rel. error $(\beta_c-\beta_c^{\rm FSS})/\beta_c^{\rm FSS}$")
    ax.set_xlabel(r"$\alpha$")
    for i in range(n):
        for j in fss_cols:
            if not np.isnan(rel[i, j]):
                ax.text(j, i, f"{rel[i, j]:+.0%}", ha="center", va="center",
                        color="k", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rel. error")

    fig.suptitle("oSSH PCA β_c vs FSS — all 32 observables (sID-5), "
                 "ranked by RMS rel. error", y=1.0, fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"pca_betac_summary.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
