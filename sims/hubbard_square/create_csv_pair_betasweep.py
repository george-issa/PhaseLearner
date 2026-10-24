"""
Convert SmoQyDQMC per-walker HDF5 bin files into PhaseLearner-ready CSVs.

The new SmoQyDQMC API stores per-bin samples in HDF5 (not CSV).  For each
(|U|, β) point this script reads all walkers, stacks their per-bin pair-pair
momentum-space correlation data, takes magnitudes (real-valued ML features),
appends β as the last column, and writes a giant CSV per |U|.

Input layout (per (|U|, β) point):
    {INPUT_PATH}/hubbard_square_U-{|U|:.4f}_n{n:.3f}_L{L}_b{β:.2f}-{sID}/
        bins/bins_pID-{0..n_walkers-1}.h5

HDF5 dataset:
    CORRELATIONS/STANDARD/EQUAL-TIME/pair/MOMENTUM
        shape = (n_id_pairs, L, L, n_bins)   complex128
        we use n_id_pairs=1 (single (1,1) on-site s-wave pair)

Output (per |U|):
    {OUTPUT_PATH}/pair_momentum_L{L}_n{n:.3f}_U{|U|:.4f}-{sID}_csv/
        pair_momentum_L{L}_n{n:.3f}_b{β:.2f}_U{|U|:.4f}-{sID}.csv  (per β)
        pair_momentum_L{L}_n{n:.3f}_U{|U|:.4f}-{sID}.csv           (concatenated)

Each row in the giant CSV: [feature_0, ..., feature_{L²-1}, β]
Matches FILE_TEMPLATE in config.py["attractive_hubbard"]:
    "pair_momentum_L8_n0.875_U{alpha:.4f}-1.csv"
"""
import os
import numpy as np
import h5py

# ── User parameters ────────────────────────────────────────────────────────────
sID = 1
L   = 8
n_target = 0.5

abs_Us = [4.0]                                # |U| values
betas  = np.linspace(2.0, 14.0, 25)           # β grid (matches the runner)

n_walkers = 2

# Where the SmoQyDQMC simulation folders live, and where to dump CSVs.
INPUT_PATH  = "data/hubbard_square/sim_data"   # SmoQyDQMC output folders
OUTPUT_PATH = "data/hubbard_square/csv"        # ML-ready CSV directories live here

# Choose observable: "pair" / "density" / "spin_z" / "greens" (greens has no EQUAL-TIME).
# Choose channel: "EQUAL-TIME" / "INTEGRATED".
# Choose space:   "MOMENTUM" / "POSITION".
OBS     = "density"
CHANNEL = "INTEGRATED"
SPACE   = "POSITION"
DATASET_PATH = f"CORRELATIONS/STANDARD/{CHANNEL}/{OBS}/{SPACE}"
SPACE_TAG    = SPACE.lower()                   # "momentum" / "position"
# Filename prefix: bare OBS for EQUAL-TIME, "<channel>_OBS" otherwise.
# This preserves backward compatibility with already-written EQ files.
CHAN_TAG     = "" if CHANNEL == "EQUAL-TIME" else CHANNEL.lower().replace("-", "_") + "_"
OBS_PREFIX   = f"{CHAN_TAG}{OBS}"               # e.g. "pair", "integrated_pair"
ID_PAIR_INDEX = 0                             # we initialized only pairs=[(1,1)]


def load_pair_momentum(sim_dir):
    """
    Read per-walker HDF5 files for one (|U|, β) point and return a
    (n_walkers * n_bins, L*L) magnitude matrix.  Skips missing walkers
    with a warning and returns None if no walker file is present.
    """
    walker_blocks = []
    for pID in range(n_walkers):
        h5path = os.path.join(sim_dir, "bins", f"bins_pID-{pID}.h5")
        if not os.path.exists(h5path):
            print(f"    walker missing: {h5path}")
            continue
        with h5py.File(h5path, "r") as f:
            arr = f[DATASET_PATH][...]                  # (1, L, L, n_bins) complex
        arr = arr[ID_PAIR_INDEX]                        # (L, L, n_bins)
        arr = np.transpose(arr, (2, 0, 1))              # (n_bins, L, L)
        arr = arr.reshape(arr.shape[0], -1)             # (n_bins, L*L)
        walker_blocks.append(np.abs(arr))               # magnitude → real features
    if not walker_blocks:
        return None
    return np.vstack(walker_blocks)                     # (n_walkers*n_bins, L*L)


for absU in abs_Us:
    print("\n" + "#" * 80)
    print(f"|U| = {absU:.4f}   (Hubbard U = -{absU:.4f})")
    print("#" * 80)

    out_dir = os.path.join(
        OUTPUT_PATH,
        f"{OBS_PREFIX}_{SPACE_TAG}_L{L}_n{n_target:.3f}_U{absU:.4f}-{sID}_csv",
    )
    os.makedirs(out_dir, exist_ok=True)

    giant_rows = []
    for b in betas:
        print("-" * 20 + f" β={b:.2f} " + "-" * 20)
        sim_dir = os.path.join(
            INPUT_PATH,
            f"hubbard_square_U-{absU:.4f}_n{n_target:.3f}_L{L}_b{b:.2f}-{sID}",
        )
        if not os.path.isdir(sim_dir):
            print(f"    not run yet: {sim_dir}")
            continue

        feats = load_pair_momentum(sim_dir)
        if feats is None:
            print(f"    no walker data — skipped")
            continue

        Sbb = np.zeros((feats.shape[0], feats.shape[1] + 1))
        Sbb[:, :-1] = feats
        Sbb[:,  -1] = b

        per_beta = (
            f"{OBS_PREFIX}_{SPACE_TAG}_L{L}_n{n_target:.3f}_b{b:.2f}_U{absU:.4f}-{sID}.csv"
        )
        np.savetxt(os.path.join(out_dir, per_beta), Sbb, delimiter=",")
        print(f"    wrote {per_beta}  shape={Sbb.shape}")
        giant_rows.append(Sbb)

    if not giant_rows:
        print(f"\n  WARN: no β data found for |U|={absU:.4f}; concat skipped")
        continue
    giant = np.vstack(giant_rows)
    giant_name = f"{OBS_PREFIX}_{SPACE_TAG}_L{L}_n{n_target:.3f}_U{absU:.4f}-{sID}.csv"
    np.savetxt(os.path.join(out_dir, giant_name), giant, delimiter=",")
    print(f"\n  giant: {giant_name}  shape={giant.shape}")
    print(
        f"  → copy/symlink to PhaseLearner data/ as "
        f"{OBS_PREFIX}_{SPACE_TAG}_L{L}_n{n_target:.3f}_U{absU:.4f}-{sID}.csv"
    )
