"""Plot an LBC accuracy sweep and locate the interior peak (estimated Tc).

Usage:  plot_lbc.py <lbc_output_dir> [output_png]

The LBC accuracy curve is a W: a trivial-accuracy arm at each end, a dip
(minimum) on either side, and a CENTRAL peak between the two dips that marks the
critical point. The naive argmax over the interior grabs the near-trivial arm,
so we locate each dip first and take the maximum strictly between them.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path(sys.argv[1])
png = Path(sys.argv[2]) if len(sys.argv) > 2 else out_dir / "lbc_ising_L4_Wcurve.png"
png.parent.mkdir(parents=True, exist_ok=True)

accs = np.loadtxt(next(out_dir.glob("accs_*.csv")))
grid = np.loadtxt(next(out_dir.glob("tc_grid_*.csv")))
TC_EXACT = 2.0 / np.log(1 + np.sqrt(2))          # 2.269185

# Each W minimum is the argmin of its half; the estimator is the max between them.
n = len(accs)
mid = n // 2
i_left_dip = 1 + np.argmin(accs[1:mid])
i_right_dip = mid + np.argmin(accs[mid:-1])
i_peak = i_left_dip + np.argmax(accs[i_left_dip:i_right_dip + 1])
tc_peak = grid[i_peak]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(grid, accs, "o-", color="#2166ac", lw=1.8, ms=5, label="LBC test accuracy")
ax.axvline(TC_EXACT, color="crimson", ls="--", lw=1.5,
           label=f"exact $T_c$ = {TC_EXACT:.3f}")
ax.axvline(tc_peak, color="green", ls=":", lw=1.5,
           label=f"LBC peak = {tc_peak:.3f}")
ax.plot(tc_peak, accs[i_peak], "*", color="green", ms=16, zorder=5)
ax.plot(grid[[i_left_dip, i_right_dip]], accs[[i_left_dip, i_right_dip]],
        "v", color="gray", ms=8, zorder=4, label="W minima")
ax.set_xlabel("proposed critical temperature  $T_c'$  (J)")
ax.set_ylabel("classifier accuracy")
ax.set_title("Learning-by-confusion — 2D Ising, L=4")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(png, dpi=150)

print(f"exact Tc          = {TC_EXACT:.4f}")
print(f"interior LBC peak = {tc_peak:.4f}  (acc={accs[i_peak]:.4f})")
print(f"|peak - Tc|       = {abs(tc_peak - TC_EXACT):.4f}")
print(f"\nWrote {png}")
