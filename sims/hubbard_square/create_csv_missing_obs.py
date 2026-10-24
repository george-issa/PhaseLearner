"""
One-shot fill-in script: extract the two BKT-relevant correlation observables
that were skipped in earlier passes, so all 8 channels exist as ML-ready CSVs.

Produces:
  integrated_density_momentum_L8_n0.875_U4.0000-1_csv/
  integrated_pair_position_L8_n0.875_U4.0000-1_csv/

Same row layout as the other create_csv_*.py scripts: each row is
[|features|..., beta]; magnitudes are taken so features are real.
"""
import os
import numpy as np
import h5py

sID, L, n_target = 1, 8, 0.875
abs_Us = [4.0]
betas  = np.linspace(2.0, 14.0, 25)
n_walkers = 4

INPUT_PATH  = "data/hubbard_square"
OUTPUT_PATH = "data/csv"
ID_PAIR_INDEX = 0

# (obs, channel, space, output_prefix)
TARGETS = [
    ("density", "INTEGRATED", "MOMENTUM", "integrated_density_momentum"),
    ("pair",    "INTEGRATED", "POSITION", "integrated_pair_position"),
]


def load_block(sim_dir, dataset_path):
    walker_blocks = []
    for pID in range(n_walkers):
        h5path = os.path.join(sim_dir, "bins", f"bins_pID-{pID}.h5")
        if not os.path.exists(h5path):
            print(f"    walker missing: {h5path}")
            continue
        with h5py.File(h5path, "r") as f:
            arr = f[dataset_path][...]
        arr = arr[ID_PAIR_INDEX]
        arr = np.transpose(arr, (2, 0, 1))
        arr = arr.reshape(arr.shape[0], -1)
        walker_blocks.append(np.abs(arr))
    if not walker_blocks:
        return None
    return np.vstack(walker_blocks)


for obs, channel, space, prefix in TARGETS:
    dataset_path = f"CORRELATIONS/STANDARD/{channel}/{obs}/{space}"
    space_tag = space.lower()
    print("\n" + "#" * 80)
    print(f"{prefix}  ←  {dataset_path}")
    print("#" * 80)

    for absU in abs_Us:
        out_dir = os.path.join(
            OUTPUT_PATH,
            f"{prefix}_L{L}_n{n_target:.3f}_U{absU:.4f}-{sID}_csv",
        )
        os.makedirs(out_dir, exist_ok=True)

        giant_rows = []
        for b in betas:
            sim_dir = os.path.join(
                INPUT_PATH,
                f"hubbard_square_U-{absU:.4f}_n{n_target:.3f}_L{L}_b{b:.2f}-{sID}",
            )
            if not os.path.isdir(sim_dir):
                print(f"  β={b:.2f}: missing simulation folder, skipped")
                continue

            feats = load_block(sim_dir, dataset_path)
            if feats is None:
                print(f"  β={b:.2f}: no walker data, skipped")
                continue

            Sbb = np.zeros((feats.shape[0], feats.shape[1] + 1))
            Sbb[:, :-1] = feats
            Sbb[:,  -1] = b

            per_beta = (
                f"{prefix}_L{L}_n{n_target:.3f}_b{b:.2f}_U{absU:.4f}-{sID}.csv"
            )
            np.savetxt(os.path.join(out_dir, per_beta), Sbb, delimiter=",")
            giant_rows.append(Sbb)

        giant = np.vstack(giant_rows)
        giant_name = f"{prefix}_L{L}_n{n_target:.3f}_U{absU:.4f}-{sID}.csv"
        np.savetxt(os.path.join(out_dir, giant_name), giant, delimiter=",")
        print(f"  giant: {giant_name}  shape={giant.shape}")
