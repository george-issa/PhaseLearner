"""
Consolidate per-(alpha, beta) phonon output files into ML-ready CSVs.

The Julia simulation writes two files inside each simulation folder
(ossh_honeycomb_w{w}_a{a}_m{m}_L{L}_b{b:.2f}-{sID}/Phonon_positions/):

  phonon_config_sID{sID}_pID0.csv
      N_bins rows × 577 cols.
      Each row = one bin snapshot of the tau=0 phonon field followed by β.
      Column layout: [Xa[0..N-1], Ya[0..N-1], Xb[0..N-1], Yb[0..N-1], β]
      where N = L*L unit cells (144 for L=12) and the 4 phonon modes are
      stored in MODE-MAJOR order (Julia column-major: mode index varies slowest).

  radial_displacement_sID{sID}_pID0.csv
      N_bins rows × 289 cols.
      Each row = one bin's time-averaged radial displacement per site, followed by β.
      Column layout: INTERLEAVED per unit cell, [r_A[0], r_B[0], r_A[1], r_B[1], ..., β]
      where r_A[i] = mean_τ sqrt(Xa[i,τ]²+Ya[i,τ]²) for sublattice-A site i,
      and   r_B[i] = mean_τ sqrt(Xb[i,τ]²+Yb[i,τ]²) for sublattice-B site i.
      NOTE: the Julia writer's own comment claims a block layout
      [A sites..., B sites...] — that is WRONG.  eval_radial_displacement fills
      R via a (2, N) column-major reshape view, so the flat vector interleaves
      A/B per unit cell.  Verified empirically 2026-06-10 by correlating the
      radial row against sqrt(Xa²+Ya²) from the τ=0 snapshot of the same bin:
      interleaved decoding corr ≈ 0.41, block decoding corr ≈ 0.00.
      This is DOMAIN-INVARIANT (magnitudes only) → recommended for PCA/UMAP.
      (Feature ordering is irrelevant for PCA/UMAP; it only matters for
      spatial visualization via decode_radial_row.)

This script stacks all beta values for each alpha into two final CSVs:

  data/phonon_tau0_L12_w0.5_m0.0_a{alpha:.4f}-1.csv   (576 + 1 cols)
  data/radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-1.csv   (288 + 1 cols)

Both have exactly the same row structure as the BondBond/greens CSVs so the
ML pipeline (data.py / pca_method.py / umap_method.py) can read them directly.

Memory layout note (tau=0 phonon file)
--------------------------------------
Julia stores the phonon array x as (Nph=576, Lτ) in column-major order.
reshape(x, (L, L, 4, Lτ)) has the first index (n1) varying fastest, then n2,
then phonon mode (1=Xa, 2=Ya, 3=Xb, 4=Yb), then τ.

So the 576-value tau=0 row is laid out as:
  [Xa(n1=0,n2=0), Xa(n1=1,n2=0), ..., Xa(n1=L-1,n2=L-1),
   Ya(n1=0,n2=0), ..., Ya(n1=L-1,n2=L-1),
   Xb(...), Yb(...)]

To decode in Python (0-indexed, N = L*L = 144):
  X_modes = row[:-1].reshape(4, N)   # (4, 144): rows are Xa, Ya, Xb, Yb
  Xa = X_modes[0].reshape(L, L, order='F')  # F-order matches Julia column-major
  Ya = X_modes[1].reshape(L, L, order='F')
  Xb = X_modes[2].reshape(L, L, order='F')
  Yb = X_modes[3].reshape(L, L, order='F')

Usage
-----
  python3 sims/ssh/extract_phonons_edited.py                # default parameters
  python3 sims/ssh/extract_phonons_edited.py --data-root /path/to/sim/folders
"""
import argparse
import os
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
W         = 0.5
M         = 0.0
L         = 12
SID       = 5       # current sweep: the 2026 sID-5 1000-bin runs
N         = L * L   # number of unit cells (144)
NPH       = 4       # phonon modes (Xa, Ya, Xb, Yb)
N_WALKERS = 4       # must match N_WALKERS in run_SSH_edited2.py

# (alpha, beta) pairs are DISCOVERED from the sim folders present on disk
# (discover_alpha_beta below) rather than hardcoded, so grid gaps — e.g. the
# 7 missing sID-5 points — are skipped automatically.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT_DEFAULT = os.path.join(REPO_ROOT, "data", "ssh", "sim_data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "ssh", "csv")


def discover_alpha_beta(data_root):
    """Map alpha -> sorted list of betas, from sim folder names on disk."""
    import re
    pat = re.compile(
        rf"ossh_honeycomb_w{W:.1f}_a([0-9.]+)_m{M:.1f}_L{L}_b([0-9.]+)-{SID}$")
    found = {}
    for name in sorted(os.listdir(data_root)):
        m = pat.match(name)
        if m:
            found.setdefault(float(m.group(1)), []).append(float(m.group(2)))
    return {a: sorted(bs) for a, bs in sorted(found.items())}


def sim_folder(data_root, a, b):
    """Return the path to one simulation folder."""
    return os.path.join(data_root,
                        f"ossh_honeycomb_w{W:.1f}_a{a:.4f}_m{M:.1f}_L{L}_b{b:.2f}-{SID}")


def load_phonon_bin(folder, file_type="phonon", n_walkers=N_WALKERS):
    """
    Load and stack the per-bin CSVs from all walkers (pID 0 … n_walkers-1).

    Each walker writes its own file:
      Phonon_positions/phonon_config_sID{SID}_pID{pID}.csv
      Phonon_positions/radial_displacement_sID{SID}_pID{pID}.csv

    All walker files are stacked row-wise so the returned array has
    n_walkers × N_bins_per_walker rows — equivalent to a single long chain.

    Parameters
    ----------
    folder : str
        Simulation data folder (contains Phonon_positions/).
    file_type : "phonon" | "radial"
        Which file to load.
    n_walkers : int
        Number of MPI walkers (must match run_SSH_edited.py N_WALKERS).

    Returns
    -------
    data : ndarray, shape (n_walkers * N_bins_per_walker, n_features+1)
        Stacked rows from all walkers. Returns None if no files are found.
    """
    ph_dir = os.path.join(folder, "Phonon_positions")
    chunks = []

    for pid in range(n_walkers):
        if file_type == "phonon":
            fname = os.path.join(ph_dir, f"phonon_config_sID{SID}_pID{pid}.csv")
        else:
            fname = os.path.join(ph_dir, f"radial_displacement_sID{SID}_pID{pid}.csv")

        if os.path.isfile(fname):
            chunks.append(np.loadtxt(fname, delimiter=","))

    if not chunks:
        return None

    return np.vstack(chunks)


def consolidate_alpha(data_root, alpha, output_dir, n_walkers=N_WALKERS,
                      verbose=True, betas=None):
    """
    Stack all beta values for one alpha into two ML-ready CSV files.

    Output files (written to output_dir):
      phonon_tau0_L12_w0.5_m0.0_a{alpha:.4f}-{SID}.csv  (576+1 cols)
      radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-{SID}.csv  (288+1 cols)
    """
    if betas is None:
        betas = discover_alpha_beta(data_root).get(alpha)
    if betas is None:
        raise ValueError(f"No sim folders found for alpha={alpha:.4f} "
                         f"under {data_root}.")

    ph_rows  = []   # tau=0 phonon field
    rd_rows  = []   # radial displacement

    missing_ph = 0
    missing_rd = 0

    for b in betas:
        folder = sim_folder(data_root, alpha, b)
        if not os.path.isdir(folder):
            if verbose:
                print(f"  [skip] folder not found: {folder}")
            missing_ph += 1
            missing_rd += 1
            continue

        ph_data = load_phonon_bin(folder, "phonon", n_walkers=n_walkers)
        rd_data = load_phonon_bin(folder, "radial", n_walkers=n_walkers)

        if ph_data is None:
            missing_ph += 1
        else:
            ph_rows.append(ph_data)

        if rd_data is None:
            missing_rd += 1
        else:
            rd_rows.append(rd_data)

    os.makedirs(output_dir, exist_ok=True)

    if ph_rows:
        ph_all = np.vstack(ph_rows)   # (N_bins_total, 577)
        ph_out = os.path.join(output_dir,
                              f"phonon_tau0_L{L}_w{W:.1f}_m{M:.1f}_a{alpha:.4f}-{SID}.csv")
        np.savetxt(ph_out, ph_all, delimiter=",", fmt="%.10f")
        if verbose:
            print(f"  Saved phonon tau0:   {ph_out}  "
                  f"({ph_all.shape[0]} rows × {ph_all.shape[1]} cols)")
    else:
        if verbose:
            print(f"  [warn] No phonon tau0 data found for alpha={alpha:.4f}")

    if rd_rows:
        rd_all = np.vstack(rd_rows)   # (N_bins_total, 289)
        rd_out = os.path.join(output_dir,
                              f"radial_disp_L{L}_w{W:.1f}_m{M:.1f}_a{alpha:.4f}-{SID}.csv")
        np.savetxt(rd_out, rd_all, delimiter=",", fmt="%.10f")
        if verbose:
            print(f"  Saved radial disp:   {rd_out}  "
                  f"({rd_all.shape[0]} rows × {rd_all.shape[1]} cols)")
    else:
        if verbose:
            print(f"  [warn] No radial displacement data found for alpha={alpha:.4f}")

    if verbose and (missing_ph or missing_rd):
        print(f"  Missing: phonon={missing_ph}, radial={missing_rd} beta points "
              f"(out of {len(betas)})")


def decode_phonon_row(row, L=12):
    """
    Decode a single tau=0 phonon row (576 values) into per-mode 2D arrays.

    Returns a dict with keys 'Xa', 'Ya', 'Xb', 'Yb', each shape (L, L).
    The reshape uses Fortran (column-major) order to match Julia's memory layout.
    """
    N = L * L
    modes = row[:4 * N].reshape(4, N)   # (4, 144): mode-major
    return {
        "Xa": modes[0].reshape(L, L, order='F'),
        "Ya": modes[1].reshape(L, L, order='F'),
        "Xb": modes[2].reshape(L, L, order='F'),
        "Yb": modes[3].reshape(L, L, order='F'),
    }


def decode_radial_row(row, L=12):
    """
    Decode a radial displacement row (288 values) into two (L, L) arrays.

    The flat row interleaves sublattices per unit cell: [A0, B0, A1, B1, ...]
    (column-major (2, N) reshape in the Julia writer — its block-layout
    comment is wrong; see module docstring for the empirical verification).

    Returns a dict with keys 'rA', 'rB', each shape (L, L).
    """
    N = L * L
    rA = row[0:2*N:2].reshape(L, L, order='F')
    rB = row[1:2*N:2].reshape(L, L, order='F')
    return {"rA": rA, "rB": rB}


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate per-(alpha,beta) phonon CSVs into ML-ready files.")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT_DEFAULT,
                        help="Directory containing simulation output folders "
                             f"(default: {DATA_ROOT_DEFAULT})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory for consolidated output CSVs "
                             f"(default: {OUTPUT_DIR})")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Subset of alpha values to process "
                             "(default: all discovered under data-root)")
    parser.add_argument("--n-walkers", type=int, default=N_WALKERS,
                        help=f"Number of MPI walkers per job (default: {N_WALKERS}). "
                             "Must match N_WALKERS in run_SSH_edited2.py.")
    args = parser.parse_args()

    grid = discover_alpha_beta(args.data_root)
    alphas = args.alphas if args.alphas is not None else list(grid)

    print(f"Data root : {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Alphas    : {alphas}")
    print(f"Walkers   : {args.n_walkers} (stacking pID 0 … {args.n_walkers - 1})")
    print()

    for alpha in alphas:
        betas = grid.get(alpha)
        print(f"alpha = {alpha:.4f}  "
              f"({len(betas) if betas else 0} beta points on disk)")
        consolidate_alpha(args.data_root, alpha, args.output_dir,
                          n_walkers=args.n_walkers, betas=betas)
        print()

    print("Done. FILE_TEMPLATE values for ssh_ml_phase/config.py:")
    print(f'  phonon tau=0 field : "phonon_tau0_L12_w0.5_m0.0_a{{alpha:.4f}}-{SID}.csv"')
    print(f'  radial displacement: "radial_disp_L12_w0.5_m0.0_a{{alpha:.4f}}-{SID}.csv"')


if __name__ == "__main__":
    main()
