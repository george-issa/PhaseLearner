#!/usr/bin/env python3
"""
Demonstrate that the Kekule registry drifts WITHIN a run at fixed (alpha, beta).

Finite-volume Gibbs averages are exactly symmetric: at fixed parameters the
Markov chain visits all three (Z3-degenerate) Kekule registries, so one-point
functions average to uniform maps even though every configuration is ordered.
This script shows it directly, deep in the ordered phase (alpha=0.2400,
beta=30 by default):

  (a) arg(rho_KVBS) vs sweep index for each walker - the phase random-walks
      through the full circle instead of locking;
  (b) the same trajectory in the complex plane (one walker, colored by MC
      time): a ring, not a lobe;
  (c) the tau=0 Kekule amplitude rho_tau0 computed from each of the 250
      phonon-config bin snapshots of one walker: every bin has |rho_tau0|
      comparable to <|rho|>, with bin-to-bin random phase; their vector mean
      (cross) is an order of magnitude smaller.

rho_tau0 is Eq. (1) of paper/notes/dqmc_observable_visualizations.tex
evaluated on the tau=0 slice instead of the tau-average:
bond stretches e_hat(theta_d).(u_B - u_A) for the three NN bonds
(theta_d = 0, -2pi/3, +2pi/3), weighted by e^{i theta_d} e^{-iK.R}.

Output: data/ssh/figures/kvbs_registry_drift_a{alpha}_b{beta}.png

Run:  python3 scripts/plot_kvbs_registry_drift.py [alpha beta]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.join(PROJECT_ROOT, "data", "ssh", "sim_data")
FIG_DIR = os.path.join(PROJECT_ROOT, "data", "ssh", "figures")
L = 12
N = L * L


def rho_tau0_per_bin(run_dir, pid=0):
    """Kekule order parameter evaluated on each bin's tau=0 phonon snapshot."""
    path = os.path.join(run_dir, "Phonon_positions",
                        f"phonon_config_sID5_pID{pid}.csv")
    ph = np.loadtxt(path, delimiter=",")[:, :-1]
    Xa, Ya, Xb, Yb = [ph[:, k * N:(k + 1) * N] for k in range(4)]

    n1, n2 = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    i0 = (n1 + L * n2).ravel()                  # F-order index of cell (n1,n2)
    nbr = [i0,                                   # bond 3: B in same cell
           (((n1 - 1) % L) + L * n2).ravel(),    # bond 4: B in cell (n1-1,n2)
           (n1 + L * ((n2 - 1) % L)).ravel()]    # bond 5: B in cell (n1,n2-1)
    theta = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])
    phase = np.exp(-1j * 2 * np.pi * (n1 + 2 * n2).ravel() / 3)

    rho = np.zeros(ph.shape[0], dtype=complex)
    for d, ib in enumerate(nbr):
        stretch = (np.cos(theta[d]) * (Xb[:, ib] - Xa[:, i0])
                   + np.sin(theta[d]) * (Yb[:, ib] - Ya[:, i0]))
        rho += np.exp(1j * theta[d]) * (stretch @ phase)
    return -rho / N


def phase_autocorr_time(rho):
    """Sweeps until |<e^{i theta(0)} e^{-i theta(t)}>| drops below 1/e."""
    z = np.exp(1j * np.angle(rho))
    z = z - z.mean()
    n = len(z)
    f = np.fft.fft(z, 2 * n)
    ac = np.fft.ifft(f * np.conj(f))[:n].real
    ac /= ac[0]
    below = np.nonzero(ac < 1 / np.e)[0]
    return int(below[0]) if below.size else n


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 2 else 0.2400
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    run = os.path.join(
        SIM_ROOT, f"ossh_honeycomb_w0.5_a{alpha:.4f}_m0.0_L12_b{beta:.2f}-5")
    os.makedirs(FIG_DIR, exist_ok=True)

    series = []
    for pid in range(4):
        d = np.loadtxt(os.path.join(
            run, "kvbs_order_parameter",
            f"kvbs_order_parameter_sID-5_pID-{pid}.txt"),
            delimiter=",", skiprows=1)
        series.append(d[:, 0] + 1j * d[:, 1])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    fig.suptitle(
        rf"Kekul\'e registry drift within single runs — "
        rf"$\alpha$={alpha:.4f}, $\beta$={beta:g} (deep ordered phase)"
        .replace(r"\'e", "é"), fontsize=11)

    # (a) accumulated phase winding vs MC time, all walkers
    ax = axes[0]
    for pid, s in enumerate(series):
        # smooth over ~1/4 of the decorrelation time before unwrapping so
        # measurement noise does not produce spurious windings
        z = np.convolve(s, np.ones(75) / 75, mode="valid")
        ax.plot(np.unwrap(np.angle(z)) / (2 * np.pi), lw=0.9,
                label=f"walker {pid}")
        tau = phase_autocorr_time(s)
        print(f"  pID{pid}: <|rho|>={np.abs(s).mean():.3f}  "
              f"|<rho>|={np.abs(s.mean()):.3f}  "
              f"phase 1/e decorrelation ~{tau} sweeps")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("sweep")
    ax.set_ylabel(r"winding of arg $\rho_{\rm KVBS}$ (turns)")
    ax.set_title("accumulated phase winding (4 walkers)", fontsize=9)
    ax.legend(fontsize=7)

    # (b) complex-plane trajectory, walker 0
    ax = axes[1]
    s = series[0][::10]
    sc = ax.scatter(s.real, s.imag, c=np.arange(len(s)), cmap="viridis",
                    s=1.5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="sweep / 10")
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re $\rho$")
    ax.set_ylabel(r"Im $\rho$")
    ax.set_title("trajectory, walker 0", fontsize=9)

    # (c) tau=0 amplitude per bin, walker 0
    ax = axes[2]
    r0 = rho_tau0_per_bin(run, pid=0)
    sc = ax.scatter(r0.real, r0.imag, c=np.arange(len(r0)), cmap="viridis",
                    s=14, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="bin index")
    m = r0.mean()
    ax.plot(m.real, m.imag, "rx", ms=12, mew=2.5,
            label=rf"bin mean ($|{{\cdot}}|$={abs(m):.2f})")
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re $\rho_{\tau=0}$")
    ax.set_ylabel(r"Im $\rho_{\tau=0}$")
    ax.set_title(rf"$\tau$=0 snapshot amplitude per bin, walker 0 "
                 rf"($\langle|\rho_{{\tau=0}}|\rangle$={np.abs(r0).mean():.2f})",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    print(f"  tau0 bins, pID0: <|rho_tau0|>={np.abs(r0).mean():.3f}  "
          f"|mean|={abs(m):.3f}")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG_DIR, f"kvbs_registry_drift_a{alpha:.4f}_b{beta:g}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
