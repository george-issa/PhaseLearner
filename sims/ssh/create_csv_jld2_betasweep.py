#!/usr/bin/env python3
"""
Extract per-bin observable CSVs from the recent (sID-5) oSSH JLD2 sim data.

The 2026 oSSH runs (data/ssh/sim_data/ossh_*-5) store equal-time correlations
as one JLD2 file per bin (equal-time/<corr>/<space>/bin-<n>_pID-<p>.jld2)
instead of the old aggregated *_bins.csv, so the old create_csv_* scripts do
not apply.  JLD2 is HDF5 underneath: each file holds `id_pairs` and a list of
(12, 12) re/im arrays, one per orbital/bond pair.

Observables produced (matching the historical pipeline conventions):

* BondBond012345_momentum  — KVBS linear combination of the 6 bond-pair
  correlators.  Old convention (create_csv_bondbond_betasweep.py, pairs
  ordered S33,S44,S55,S43,S54,S53 with im[S53] sign-flipped):
      K = sum(re_diag) - sum(re_off) - sqrt(3)*(im43 + im54 - im53)
  The new JLD2 files store the transposed cross pairs (3,4),(3,5),(4,5).
  Under hermiticity C_ab(q) = conj(C_ba(q)) the same observable reads
      K = sum(re_diag) - (re34 + re35 + re45) + sqrt(3)*(im34 + im45 - im35)
  Set BOND_IM_SIGN = -1 to flip the sqrt(3) term if validation against the
  old -1 CSVs disagrees (see --validate).

* greens0_momentum — |C_(1,1)(q)| magnitudes (orbital-1 Green's function).

Output (pipeline-ready, rows = [144 features..., beta]):
    data/ssh/csv/BondBond012345_momentum_L12_w0.5_m0.0_a{alpha:.4f}-5.csv
    data/ssh/csv/greens0_momentum_L12_w0.5_m0.0_a{alpha:.4f}-5.csv

Run:  python sims/ssh/create_csv_jld2_betasweep.py [--workers 8] [--validate]
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "data" / "ssh" / "sim_data"
OUT_DIR = ROOT / "data" / "ssh" / "csv"
RUN_RE = re.compile(
    r"ossh_honeycomb_w0\.5_a(?P<alpha>[0-9.]+)_m0\.0_L12_b(?P<beta>[0-9.]+)-5$"
)

N_WALKERS = 4
N_BINS = 250
SQ3 = np.sqrt(3.0)
BOND_IM_SIGN = +1  # sign of the sqrt(3) im term; validated vs old -1 CSVs


def _read_pairs(path):
    """All correlation pair arrays of one JLD2 bin file, (n_pairs, 144) complex."""
    with h5py.File(path, "r") as f:
        refs = f["correlations"][:]
        out = np.empty((len(refs), 144), dtype=complex)
        for k, ref in enumerate(refs):
            arr = f[ref][:]
            out[k] = (arr["re"] + 1j * arr["im"]).reshape(144)
    return out


def _bond_kvbs(C):
    """KVBS combination from pairs ordered (3,3),(4,4),(5,5),(3,4),(3,5),(4,5)."""
    re_part = C.real
    im_part = C.imag
    k = (re_part[0] + re_part[1] + re_part[2]
         - (re_part[3] + re_part[4] + re_part[5])
         + BOND_IM_SIGN * SQ3 * (im_part[3] + im_part[5] - im_part[4]))
    return k


def reduce_run(run_dir):
    """One (alpha, beta) run -> dict of observable -> (n_samples, 145) block."""
    m = RUN_RE.match(run_dir.name)
    alpha, beta = float(m["alpha"]), float(m["beta"])
    blocks = {}
    spec = {
        "BondBond012345_momentum": (run_dir / "equal-time" / "bond" / "momentum",
                                    _bond_kvbs),
        "greens0_momentum": (run_dir / "equal-time" / "greens" / "momentum",
                             lambda C: np.abs(C[0])),
    }
    for obs, (folder, combine) in spec.items():
        rows, missing = [], 0
        for pid in range(N_WALKERS):
            for b in range(1, N_BINS + 1):
                path = folder / f"bin-{b}_pID-{pid}.jld2"
                if not path.exists():
                    missing += 1
                    continue
                rows.append(combine(_read_pairs(path)))
        if not rows:
            print(f"  WARNING: no {obs} bins in {run_dir.name}", flush=True)
            continue
        if missing:
            print(f"  WARNING: {missing} missing {obs} bins in {run_dir.name}",
                  flush=True)
        block = np.empty((len(rows), 145))
        block[:, :-1] = np.asarray(rows)
        block[:, -1] = beta
        blocks[obs] = block
    return alpha, beta, blocks


def validate(table_by_alpha_obs):
    """Compare new extraction vs old -1 CSVs at matching (alpha, beta).

    Permutation-invariant checks (feature ordering differs between the old
    bins-CSV export and the JLD2 layout): per-sample feature mean, and the
    sorted per-feature means.
    """
    for obs, template in [
        ("BondBond012345_momentum", "BondBond012345_momentum_L12_w0.5_m0.0_a{a:.4f}-1.csv"),
        ("greens0_momentum", "greens0_momentum_L12_w0.5_m0.0_a{a:.4f}-1.csv"),
    ]:
        for a in (0.2400, 0.2500):
            old_f = OUT_DIR / template.format(a=a)
            if not old_f.exists() or (a, obs) not in table_by_alpha_obs:
                continue
            old = np.loadtxt(old_f, delimiter=",")
            new = table_by_alpha_obs[(a, obs)]
            for b in (10.0, 20.0):
                o = old[np.isclose(old[:, -1], b)][:, :-1]
                n = new[np.isclose(new[:, -1], b)][:, :-1]
                if len(o) == 0 or len(n) == 0:
                    continue
                om, nm = o.mean(), n.mean()
                osd = o.mean(axis=1).std()
                print(f"[validate] {obs} a={a} b={b}: "
                      f"old mean={om:.4f} new mean={nm:.4f} "
                      f"(old sample-sd {osd:.4f}, n_old={len(o)}, n_new={len(n)}) "
                      f"sorted-feature-mean max|diff|="
                      f"{np.max(np.abs(np.sort(o.mean(0)) - np.sort(n.mean(0)))):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--validate", action="store_true",
                    help="compare against old -1 CSVs at matching (alpha, beta)")
    args = ap.parse_args()

    runs = sorted(d for d in SIM_ROOT.iterdir() if d.is_dir() and RUN_RE.match(d.name))
    print(f"{len(runs)} sID-5 runs found")

    with Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(reduce_run, runs), 1):
            results.append(res)
            print(f"\r[{i}/{len(runs)}] a={res[0]:.4f} b={res[1]:5.2f}",
                  end="", flush=True)
    print()

    # group blocks per (alpha, observable), ordered by beta
    grouped = defaultdict(list)
    for alpha, beta, blocks in results:
        for obs, block in blocks.items():
            grouped[(alpha, obs)].append((beta, block))

    table_by_alpha_obs = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for (alpha, obs), items in sorted(grouped.items()):
        items.sort(key=lambda t: t[0])
        giant = np.vstack([blk for _, blk in items])
        table_by_alpha_obs[(alpha, obs)] = giant
        out = OUT_DIR / f"{obs}_L12_w0.5_m0.0_a{alpha:.4f}-5.csv"
        np.savetxt(out, giant, delimiter=",")
        print(f"wrote {out.name}  shape={giant.shape}  "
              f"betas={items[0][0]:.1f}..{items[-1][0]:.1f} ({len(items)})")

    if args.validate:
        validate(table_by_alpha_obs)


if __name__ == "__main__":
    main()
