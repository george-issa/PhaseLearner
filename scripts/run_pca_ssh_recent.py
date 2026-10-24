#!/usr/bin/env python3
"""
PCA beta_c recovery on the recent (sID-5) oSSH runs.

Inputs:  data/ssh/csv/<obs>_L12_w0.5_m0.0_a{alpha:.4f}-5.csv
         (built by create_csv_jld2_betasweep.py from the JLD2 sim data)
Outputs: results/npz/pca_results_<obs>_sid5.npz
         results/figures/ssh/pca/alpha<a>/<obs>/results.pdf
         (merged avg_analysis + embedding + PC_layout pages for that alpha;
          the three component PDFs are kept alongside)

Run:  python scripts/run_pca_ssh_recent.py [observable ...]
      (no args = all of OBSERVABLES_DEFAULT)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run
from ssh_ml_phase import config as _cfg

ALPHAS = [0.240, 0.2425, 0.245, 0.250, 0.255]
OBSERVABLES_DEFAULT = ["BondBond012345_momentum", "greens0_momentum",
                       "phonon_tau0", "radial_disp"]
OBSERVABLES = sys.argv[1:] or OBSERVABLES_DEFAULT
GS = "/opt/homebrew/bin/gs"

BETA_MIN = None  # set e.g. 8.0 to cut the low-beta V-shape artifact region
BETA_MAX = None


def merge_pdfs(parts, out):
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        return
    subprocess.run(
        [GS, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
         "-sDEVICE=pdfwrite", f"-sOutputFile={out}", *parts],
        check=True,
    )
    print(f"  merged -> {out}")


def main():
    from ssh_ml_phase.plot import plot_pca_beta_sweep

    for obs in OBSERVABLES:
        template = f"{obs}_L12_w0.5_m0.0_a{{alpha:.4f}}-5.csv"
        tag = f"{obs}_sid5"

        run(model="ssh", alphas=ALPHAS, file_template=template, tag=tag,
            beta_min=BETA_MIN, beta_max=BETA_MAX, methods=["pca"], plot=False)

        npz = os.path.join(_cfg.NPZ_DIR, f"pca_results_{tag}.npz")
        for a in ALPHAS:
            fig_dir = os.path.join(_cfg.RESULTS_DIR, "figures", "ssh", "pca",
                                   f"alpha{a:.4f}", obs)
            os.makedirs(fig_dir, exist_ok=True)
            model_info = {"model": "ssh", "observable": obs, "L": _cfg.L,
                          "beta_min": BETA_MIN, "beta_max": BETA_MAX, "tag": tag}
            plot_pca_beta_sweep(npz, [a], _cfg.FSS_DICT, model_info, fig_dir)
            merge_pdfs(
                [os.path.join(fig_dir, n + ".pdf")
                 for n in ("avg_analysis", "embedding", "PC_layout")],
                os.path.join(fig_dir, "results.pdf"),
            )


if __name__ == "__main__":
    main()
