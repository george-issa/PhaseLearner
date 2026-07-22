"""Generate 2D Ising snapshots via checkerboard Metropolis, write an LBC CSV.

Output CSV shape (N, L*L + 1): first L*L cols are one spin config (+/-1),
last col is the temperature T. Exact critical point Tc = 2/ln(1+sqrt2) ~ 2.269.
Used to validate the LBC pipeline in lbc_train.py against a known transition.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def checkerboard_metropolis(state, T, n_sweeps, mask0, mask1, rng):
    """In-place Metropolis sweeps on a batch of lattices. state: (C,L,L) +/-1."""
    for _ in range(n_sweeps):
        for mask in (mask0, mask1):
            neigh = (np.roll(state, 1, 1) + np.roll(state, -1, 1)
                     + np.roll(state, 1, 2) + np.roll(state, -1, 2))
            dE = 2.0 * state * neigh                       # J = 1
            accept = (dE <= 0) | (rng.random(state.shape) < np.exp(-dE / T))
            flip = accept & mask
            state[flip] *= -1
    return state


def gen_for_T(T, L, n_chains, warmup, decorr, snaps_per_chain, rng):
    """Independent snapshots at temperature T from n_chains parallel lattices."""
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    mask0 = ((ii + jj) % 2 == 0)[None, :, :]
    mask1 = ~mask0
    state = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_chains, L, L))
    checkerboard_metropolis(state, T, warmup, mask0, mask1, rng)
    out = []
    for _ in range(snaps_per_chain):
        checkerboard_metropolis(state, T, decorr, mask0, mask1, rng)
        out.append(state.reshape(n_chains, -1).copy())
    return np.vstack(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_csv", type=Path)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--tmin", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=3.6)
    p.add_argument("--ntemp", type=int, default=27)
    p.add_argument("--chains", type=int, default=300)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--decorr", type=int, default=30)
    p.add_argument("--snaps", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    temps = np.round(np.linspace(a.tmin, a.tmax, a.ntemp), 4)
    rows = []
    for T in temps:
        cfgs = gen_for_T(T, a.L, a.chains, a.warmup, a.decorr, a.snaps, rng)
        m = np.abs(cfgs.mean(axis=1)).mean()          # <|m|> per spin, sanity
        block = np.concatenate([cfgs, np.full((cfgs.shape[0], 1), T)], axis=1)
        rows.append(block)
        print(f"T={T:.3f}  n={cfgs.shape[0]:5d}  <|m|>={m:.3f}")
    data = np.vstack(rows).astype(np.float64)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out_csv, data, delimiter=",")
    print(f"\nWrote {a.out_csv}  shape={data.shape}")


if __name__ == "__main__":
    main()
