#!/usr/bin/env python3
"""
Heatmap visualization of the new SSH DQMC phonon outputs.

Two observables, written by the Julia runs into each sim folder's
Phonon_positions/ and consolidated by sims/ssh/extract_phonons_edited.py into
data/ssh/csv/ (sID-5, 1000 bins x 33 betas = 33000 rows per alpha):

  phonon_tau0_L12_w0.5_m0.0_a{alpha:.4f}-5.csv          (33000, 577)
      Row = one bin's tau=0 phonon field, then beta.
      Layout: [Xa(144), Ya(144), Xb(144), Yb(144), beta] - mode-major,
      each 144-block in Julia column-major (n1 fastest), so a block
      decodes to a unit-cell grid via reshape(L, L, order='F').
      Xs/Ys = x/y displacement of the sublattice-s atom in each cell.

  radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-5.csv          (33000, 289)
      Row = one bin's tau-AVERAGED radial displacement per site, then beta:
      r_s[i] = mean_tau sqrt(Xs[i,tau]^2 + Ys[i,tau]^2),  s in {A, B}.
      Layout: interleaved per unit cell [rA0, rB0, rA1, rB1, ..., beta]
      (the Julia writer's block-layout comment is wrong; see
      sims/ssh/extract_phonons_edited.py for the empirical verification).

For each alpha, two figures are saved to data/ssh/figures/:

  phonon_tau0_heatmaps_a{alpha:.4f}.png
      rows = 4 representative betas (low, ~0.75*beta_c, ~beta_c, high);
      cols = Xa, Ya, Xb, Yb of a single bin snapshot (diverging colormap,
      symmetric color limits shared across the row).

  radial_disp_heatmaps_a{alpha:.4f}.png
      same beta rows; cols = single-bin rA, rB and the 1000-bin means
      <rA>, <rB> at that beta (the means are where the ordering pattern
      becomes visible above the snapshot noise).

Run:  python3 scripts/plot_phonon_heatmaps.py [alpha ...]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sims", "ssh"))
from extract_phonons_edited import decode_phonon_row, decode_radial_row

L = 12
SID = 5
CSV_DIR = os.path.join(PROJECT_ROOT, "data", "ssh", "csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "data", "ssh", "figures")

ALPHAS = [0.2400, 0.2425, 0.2450, 0.2500, 0.2550]

# FSS beta_c references (w=0.5); 0.2425 interpolated between neighbours.
FSS_ALPHA = np.array([0.240, 0.245, 0.250, 0.255])
FSS_BETA_C = np.array([17.5, 14.3, 12.5, 12.3])


def beta_c(alpha):
    return float(np.interp(alpha, FSS_ALPHA, FSS_BETA_C))


def load_csv(template, alpha):
    path = os.path.join(CSV_DIR, template.format(alpha=alpha))
    data = np.loadtxt(path, delimiter=",")
    X, betas = data[:, :-1], data[:, -1]
    print(f"  {os.path.basename(path)}: rows={data.shape[0]}  "
          f"features={X.shape[1]}  betas={betas.min():.1f}..{betas.max():.1f} "
          f"({len(np.unique(betas))} values)")
    return X, betas


def snapshot_betas(u_beta, bc):
    """Low beta, ~0.75*beta_c, nearest to beta_c, high beta (deduplicated)."""
    targets = [u_beta[0], 0.75 * bc, bc, u_beta[-1]]
    picks = [u_beta[np.argmin(np.abs(u_beta - t))] for t in targets]
    return list(dict.fromkeys(picks))


def plot_phonon_tau0(alpha):
    X, betas = load_csv("phonon_tau0_L12_w0.5_m0.0_a{alpha:.4f}-" f"{SID}.csv",
                        alpha)
    bc = beta_c(alpha)
    rows = snapshot_betas(np.unique(betas), bc)
    cols = ["Xa", "Ya", "Xb", "Yb"]

    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.2 * len(cols), 3.0 * len(rows)),
                             squeeze=False)
    fig.suptitle(rf"$\tau=0$ phonon field, single bin snapshot — "
                 rf"$\alpha$={alpha:.4f}, FSS $\beta_c$={bc:.1f}"
                 "\ncolumns: (x, y) displacement of sublattice A / B atoms "
                 "per unit cell", fontsize=11)

    for r, b in enumerate(rows):
        sample = X[betas == b][0]
        d = decode_phonon_row(sample, L=L)
        vmax = max(abs(d[k]).max() for k in cols)
        for c, key in enumerate(cols):
            ax = axes[r, c]
            im = ax.imshow(d[key].T, origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(rf"{key}   $\beta$={b:.1f}", fontsize=9)
            ax.set_xlabel("$n_1$" if r == len(rows) - 1 else "")
            ax.set_ylabel("$n_2$" if c == 0 else "")
            if c == len(cols) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, f"phonon_tau0_heatmaps_a{alpha:.4f}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_radial_disp(alpha):
    X, betas = load_csv("radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-" f"{SID}.csv",
                        alpha)
    bc = beta_c(alpha)
    rows = snapshot_betas(np.unique(betas), bc)

    fig, axes = plt.subplots(len(rows), 4,
                             figsize=(3.4 * 4, 3.0 * len(rows)),
                             squeeze=False)
    fig.suptitle(rf"$\tau$-averaged radial displacement — "
                 rf"$\alpha$={alpha:.4f}, FSS $\beta_c$={bc:.1f}"
                 "\ncolumns: single bin rA, rB | 1000-bin means "
                 r"$\langle rA\rangle$, $\langle rB\rangle$", fontsize=11)

    for r, b in enumerate(rows):
        Xb_ = X[betas == b]
        sample = decode_radial_row(Xb_[0], L=L)
        mean = decode_radial_row(Xb_.mean(axis=0), L=L)
        panels = [(sample["rA"], "rA (1 bin)"), (sample["rB"], "rB (1 bin)"),
                  (mean["rA"], r"$\langle rA\rangle$ (1000 bins)"),
                  (mean["rB"], r"$\langle rB\rangle$ (1000 bins)")]
        s_lim = (min(p[0].min() for p in panels[:2]),
                 max(p[0].max() for p in panels[:2]))
        m_lim = (min(p[0].min() for p in panels[2:]),
                 max(p[0].max() for p in panels[2:]))
        for c, (img, label) in enumerate(panels):
            ax = axes[r, c]
            vmin, vmax = s_lim if c < 2 else m_lim
            im = ax.imshow(img.T, origin="lower", cmap="hot",
                           vmin=vmin, vmax=vmax)
            ax.set_title(rf"{label}   $\beta$={b:.1f}", fontsize=9)
            ax.set_xlabel("$n_1$" if r == len(rows) - 1 else "")
            ax.set_ylabel("$n_2$" if c == 0 else "")
            if c in (1, 3):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, f"radial_disp_heatmaps_a{alpha:.4f}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    alphas = [float(a) for a in sys.argv[1:]] or ALPHAS
    os.makedirs(FIG_DIR, exist_ok=True)
    for alpha in alphas:
        print(f"alpha = {alpha:.4f}")
        plot_phonon_tau0(alpha)
        plot_radial_disp(alpha)


if __name__ == "__main__":
    main()
