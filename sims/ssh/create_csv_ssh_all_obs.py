#!/usr/bin/env python3
"""
Extract per-bin CSVs for *all* oSSH correlation observables from the recent
(sID-5) JLD2 sim data, one CSV per (observable, alpha).

Background
----------
The 2026 oSSH runs (data/ssh/sim_data/ossh_*-5) store correlations as one JLD2
file per bin under

    <run>/<channel>/<corr>/<space>/bin-<n>_pID-<p>.jld2

with channel ∈ {equal-time, integrated}, corr ∈ {bond, greens},
space ∈ {momentum, position}.  JLD2 is HDF5 underneath: each file holds
`id_pairs` and a list of (12, 12) re/im arrays, one per orbital/bond pair.

This generalises create_csv_jld2_betasweep.py (which only emitted the two
equal-time KVBS/greens momentum channels) to every channel present on disk,
treating **each id-pair separately** as its own observable, plus the bond
**KVBS combination** in momentum space only.

Featurization (one rule per space, documented so it is easy to flip)
--------------------------------------------------------------------
* Momentum pairs are complex → feature = |C(q)|        (magnitude, 144 feats)
  matches the existing greens0_momentum / Hubbard np.abs convention; the
  phase information is captured separately by the KVBS combination.
* Position pairs are exactly real (Im == 0 on disk) → feature = Re C(r)
  (signed, 144 feats) — magnitude would needlessly discard the sign.
* KVBS combo (momentum only) uses the existing *signed* linear combination
  (create_csv_jld2_betasweep._bond_kvbs); its √3·Im term is only meaningful
  in momentum space, which is why it is emitted for momentum channels only.

Set POSITION_TAKE_ABS = True to featurize position pairs by magnitude instead.

Observables produced (each × every alpha on disk)
-------------------------------------------------
  bond pairs (6)        : bond{ab}_position,  bond{ab}_momentum
  integrated bond (6)   : integrated_bond{ab}_position, integrated_bond{ab}_momentum
  greens pairs (3)      : greens{ab}_position, greens{ab}_momentum     (eq-time only)
  KVBS combo            : BondBond012345_momentum,
                          integrated_BondBond012345_momentum
  → 32 observables total. {ab} = id-pair, e.g. 33, 44, 55, 34, 35, 45 / 11, 22, 12.

Output (pipeline-ready, rows = [144 features..., beta]):
    data/ssh/csv/<obs>_L12_w0.5_m0.0_a{alpha:.4f}-5.csv

Run:  python sims/ssh/create_csv_ssh_all_obs.py [--workers 8]
                                       [--alphas 0.2400 0.2500 ...]
                                       [--validate]
"""
import argparse
import os
import re
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

N_FEATURES = 144            # 12 × 12
SQ3 = np.sqrt(3.0)
BOND_IM_SIGN = +1           # sign of the √3 im term in the KVBS combo
POSITION_TAKE_ABS = False   # True → |Re C(r)| for position pairs instead of signed

# Channels present on disk: (channel_dir, corr, space).
# integrated/greens is intentionally absent (no such folder is produced).
CHANNELS = [
    ("equal-time", "bond",   "momentum"),
    ("equal-time", "bond",   "position"),
    ("equal-time", "greens", "momentum"),
    ("equal-time", "greens", "position"),
    ("integrated", "bond",   "momentum"),
    ("integrated", "bond",   "position"),
]

# Expected bond id-pair order for the KVBS combination (assert before using).
KVBS_PAIR_ORDER = [(3, 3), (4, 4), (5, 5), (3, 4), (3, 5), (4, 5)]


def _read_bin(path):
    """Return (id_pairs, C) for one JLD2 bin file.

    id_pairs : list of (a, b) int tuples, one per correlation pair.
    C        : complex ndarray, shape (n_pairs, 144).
    """
    with h5py.File(path, "r") as f:
        ids = [(int(t["1"]), int(t["2"])) for t in f["id_pairs"][:]]
        refs = f["correlations"][:]
        C = np.empty((len(refs), N_FEATURES), dtype=complex)
        for k, ref in enumerate(refs):
            arr = f[ref][:]
            C[k] = (arr["re"] + 1j * arr["im"]).reshape(N_FEATURES)
    return ids, C


def _bond_kvbs(C):
    """Signed KVBS combination from pairs ordered (3,3),(4,4),(5,5),(3,4),(3,5),(4,5).

    K = ΣRe(diag) − ΣRe(off) + sign·√3·(Im34 + Im45 − Im35).
    Mirrors create_csv_jld2_betasweep._bond_kvbs exactly.
    """
    re_part, im_part = C.real, C.imag
    return (re_part[0] + re_part[1] + re_part[2]
            - (re_part[3] + re_part[4] + re_part[5])
            + BOND_IM_SIGN * SQ3 * (im_part[3] + im_part[5] - im_part[4]))


def _pair_feature(C_pair, space):
    """Featurize one pair's (144,) complex vector for the given space."""
    if space == "momentum":
        return np.abs(C_pair)
    feat = C_pair.real
    return np.abs(feat) if POSITION_TAKE_ABS else feat


def reduce_run(run_dir):
    """One (alpha, beta) run → dict {obs_name: (n_samples, 145) block}.

    Each bin file is read exactly once and contributes to every observable
    derived from its folder (all pairs + KVBS where applicable).
    """
    m = RUN_RE.match(run_dir.name)
    alpha, beta = float(m["alpha"]), float(m["beta"])
    blocks = {}

    for channel, corr, space in CHANNELS:
        folder = run_dir / channel / corr / space
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("bin-*_pID-*.jld2"))
        if not files:
            continue

        prefix = "integrated_" if channel == "integrated" else ""
        make_kvbs = (corr == "bond" and space == "momentum")

        ids = None
        pair_rows = None          # list-of-lists, one per pair index
        kvbs_rows = []
        for fp in files:
            file_ids, C = _read_bin(fp)
            if ids is None:
                ids = file_ids
                pair_rows = [[] for _ in range(len(ids))]
            for pidx in range(C.shape[0]):
                pair_rows[pidx].append(_pair_feature(C[pidx], space))
            if make_kvbs:
                assert ids == KVBS_PAIR_ORDER, (
                    f"unexpected bond pair order {ids} in {folder}")
                kvbs_rows.append(_bond_kvbs(C))

        for pidx, (a, b) in enumerate(ids):
            obs = f"{prefix}{corr}{a}{b}_{space}"
            blocks[obs] = _stack_with_beta(pair_rows[pidx], beta)
        if make_kvbs:
            obs = f"{prefix}BondBond012345_{space}"
            blocks[obs] = _stack_with_beta(kvbs_rows, beta)

    return alpha, beta, blocks


def _stack_with_beta(rows, beta):
    block = np.empty((len(rows), N_FEATURES + 1))
    block[:, :-1] = np.asarray(rows)
    block[:, -1] = beta
    return block


def validate(table):
    """Sanity-check new KVBS/greens momentum vs the existing sID-5 CSVs
    (built by create_csv_jld2_betasweep.py) at matching (alpha, beta).

    Permutation-invariant check (feature ordering may differ): compare the
    sorted per-feature means.  KVBS is signed → compared directly; the legacy
    greens0_momentum is |C_(1,1)| → compared against new greens11_momentum.
    """
    checks = [
        ("BondBond012345_momentum",
         "BondBond012345_momentum_L12_w0.5_m0.0_a{a:.4f}-5.csv"),
        ("greens11_momentum",
         "greens0_momentum_L12_w0.5_m0.0_a{a:.4f}-5.csv"),
    ]
    for obs, template in checks:
        for a in (0.2400, 0.2500):
            old_f = OUT_DIR / template.format(a=a)
            if not old_f.exists() or (a, obs) not in table:
                continue
            old = np.loadtxt(old_f, delimiter=",")
            new = table[(a, obs)]
            for b in (10.0, 20.0):
                o = old[np.isclose(old[:, -1], b)][:, :-1]
                n = new[np.isclose(new[:, -1], b)][:, :-1]
                if len(o) == 0 or len(n) == 0:
                    continue
                d = np.max(np.abs(np.sort(o.mean(0)) - np.sort(n.mean(0))))
                print(f"[validate] {obs} a={a} b={b}: "
                      f"old mean={o.mean():.4f} new mean={n.mean():.4f} "
                      f"sorted-feature-mean max|diff|={d:.4e} "
                      f"(n_old={len(o)}, n_new={len(n)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="subset of alpha values (default: all discovered)")
    ap.add_argument("--validate", action="store_true",
                    help="compare KVBS/greens momentum vs existing sID-5 CSVs")
    args = ap.parse_args()

    runs = sorted(d for d in SIM_ROOT.iterdir()
                  if d.is_dir() and RUN_RE.match(d.name))
    if args.alphas is not None:
        keep = {f"{a:.4f}" for a in args.alphas}
        runs = [d for d in runs if RUN_RE.match(d.name)["alpha"] in keep]
    print(f"{len(runs)} sID-5 runs to process  (workers={args.workers})")

    with Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(reduce_run, runs), 1):
            results.append(res)
            print(f"\r[{i}/{len(runs)}] a={res[0]:.4f} b={res[1]:5.2f}  "
                  f"obs={len(res[2])}   ", end="", flush=True)
    print()

    # group per (alpha, observable), ordered by beta
    grouped = defaultdict(list)
    for alpha, beta, blocks in results:
        for obs, block in blocks.items():
            grouped[(alpha, obs)].append((beta, block))

    table = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_written = 0
    obs_seen = set()
    for (alpha, obs), items in sorted(grouped.items()):
        items.sort(key=lambda t: t[0])
        giant = np.vstack([blk for _, blk in items])
        table[(alpha, obs)] = giant
        out = OUT_DIR / f"{obs}_L12_w0.5_m0.0_a{alpha:.4f}-5.csv"
        np.savetxt(out, giant, delimiter=",")
        n_written += 1
        obs_seen.add(obs)
        print(f"wrote {out.name:<52s} shape={giant.shape} "
              f"betas={items[0][0]:.1f}..{items[-1][0]:.1f} ({len(items)})")

    print(f"\n{n_written} CSVs written across {len(obs_seen)} observables:")
    for o in sorted(obs_seen):
        print(f"  {o}")

    if args.validate:
        print()
        validate(table)


if __name__ == "__main__":
    main()
