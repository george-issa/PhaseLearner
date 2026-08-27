"""
Runner for the 2D attractive square Hubbard pilot.

Two modes (toggle via the LOCAL flag below):

  LOCAL = True   →  Dell-Tower / workstation: runs mpirun directly via
                    subprocess; no SLURM required.  Sequentially loops over
                    (|U|, β) points, optionally with limited parallelism.

  LOCAL = False  →  cluster: writes one SLURM .sh per (|U|, β) point and
                    submits with sbatch (mirrors run_SSH_edited2.py).

Pipeline convention:
  - ALPHAS in PhaseLearner = |U| (positive); the Julia driver flips the sign.
  - Folder produced by the driver:
      hubbard_square_U-{|U|:.4f}_n{n:.3f}_L{L}_b{β:.2f}-{sID}
"""
import argparse
import numpy as np
import subprocess
import shutil
import os
import sys

# ── Run mode ──────────────────────────────────────────────────────────────────
LOCAL = True             # True = Dell Tower (mpirun direct); False = SLURM cluster

# Concurrency cap for LOCAL mode: how many (|U|, β) points to run at once.
# Each job uses N_WALKERS cores → cap × N_WALKERS ≲ physical cores on the box.
# Default 1 keeps the box responsive; bump to 2–4 once you've timed one point.
LOCAL_PARALLEL = 8

# MPI launcher.  Set to None to auto-detect: prefer Julia's `mpiexecjl`
# (works with the JLL-bundled MPI, no system install needed), then `mpirun`.
MPI_LAUNCHER = None

# ── Output layout ──────────────────────────────────────────────────────────────
# Simulation artifacts (.jl and .log) → sims/hubbard_square/
# Data outputs (SmoQyDQMC folders)    → data/hubbard_square/sim_data/
REPO_ROOT       = os.path.dirname(os.path.abspath(__file__))
SIMULATIONS_DIR = os.path.join(REPO_ROOT, "sims", "hubbard_square")
DATA_DIR        = os.path.join(REPO_ROOT, "data", "hubbard_square", "sim_data")

source_file = os.path.join(SIMULATIONS_DIR, "attractive_hubbard_square_mpi_execute.jl")

# ── Parallel walkers ───────────────────────────────────────────────────────────
# 4 MPI walkers × 100 bins/walker = 400 samples per (|U|, β) point.
N_WALKERS         = 2
N_BINS_TOTAL      = 400
N_BINS_PER_WALKER = N_BINS_TOTAL // N_WALKERS   # 200

# ── Per-|U| beta grids ─────────────────────────────────────────────────────────
# β grid: 25 points β∈[2,14] step 0.5.  Pilot bracket is for U=-4
# (Literature T_c≈0.118 t → β_c≈8.5); widen manually for other |U|.
# |U| is set via --abs-u on the command line; default 4.0.

# ── Physical/algorithmic parameters ────────────────────────────────────────────
ABS_U    = 6.0          # |U|/t (CLI --abs-u overrides this)
N_TARGET = 0.875        # target filling (CLI --n overrides this)
MU_INIT  = 0.0          # initial chemical potential — MuTuner will adjust
L        = 8            # lattice (8×8 for pilot; rerun L=10, 12 for FSS later)


def generate_jl_script(sID, absU, n, mu, L, beta, N_therm, N_measurements,
                       N_bins, N_updates):
    """Per-point Julia wrapper that includes the driver and forwards ARGS."""
    script_content = f"""
using MPI
include("{source_file}")

MPI.Init()
comm = MPI.COMM_WORLD

sID            = parse(Int,     ARGS[1])
absU           = parse(Float64, ARGS[2])
n              = parse(Float64, ARGS[3])
μ              = parse(Float64, ARGS[4])
L              = parse(Int,     ARGS[5])
β              = parse(Float64, ARGS[6])
N_therm        = parse(Int,     ARGS[7])
N_measurements = parse(Int,     ARGS[8])
N_bins         = parse(Int,     ARGS[9])
N_updates      = parse(Int,     ARGS[10])

run_attractive_hubbard_square_simulation(
    comm;
    sID            = sID,
    U              = -absU,
    n              = n,
    μ              = μ,
    L              = L,
    β              = β,
    N_therm        = N_therm,
    N_measurements = N_measurements,
    N_bins         = N_bins,
    N_updates      = N_updates,
)

MPI.Finalize()
    """
    os.makedirs(SIMULATIONS_DIR, exist_ok=True)
    script_name = os.path.join(
        SIMULATIONS_DIR,
        f"hubbard_square_U{absU:.4f}_n{n:.3f}_L{L}_b{beta:.2f}-{sID}.jl",
    )
    with open(script_name, "w") as f:
        f.write(script_content)
    return script_name


def generate_slurm_script(sID, absU, n, mu, L, beta, N_therm, N_measurements,
                          N_bins, N_updates, n_walkers=1):
    job_name = f"hubbard_square_U{absU:.4f}_n{n:.3f}_L{L}_b{beta:.2f}-{sID}"
    jl_script  = f"{job_name}.jl"
    sh_script  = f"{job_name}.sh"

    with open(sh_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --partition=puma,puma-i9\n")
        f.write(f"#SBATCH -J {job_name}\n")
        f.write(f"#SBATCH -o {job_name}-%j.output\n")
        f.write(f"#SBATCH --ntasks={n_walkers}\n")
        f.write(f"#SBATCH --cpus-per-task=1\n")
        # f.write(f"#SBATCH --time=24:00:00\n")

        # MPI is configured system-wide on puma; srun handles task placement.
        f.write(
            f"time mpirun -n {n_walkers} julia {jl_script} "
            f"{sID} {absU} {n} {mu} {L} {beta} "
            f"{N_therm} {N_measurements} {N_bins} {N_updates}\n"
        )

    subprocess.run(["sbatch", sh_script])
    print(
        f"Submitted: |U|={absU:.4f}  β={beta:.2f}  walkers={n_walkers}  "
        f"bins/walker={N_bins}  total_samples={n_walkers * N_bins}"
    )


def _resolve_launcher():
    """Pick an MPI launcher: explicit MPI_LAUNCHER, else mpiexecjl, else mpirun."""
    if MPI_LAUNCHER:
        return MPI_LAUNCHER
    for cand in ("mpiexecjl", "mpirun"):
        if shutil.which(cand):
            return cand
    sys.exit("ERROR: no MPI launcher found (need mpiexecjl or mpirun on PATH)")


def _point_already_done(absU, n, L, beta, sID, n_walkers, min_size_bytes=1_000_000):
    """Return True if all n_walkers bins_pID-*.h5 files exist and exceed min_size."""
    sim_dir = os.path.join(
        DATA_DIR, f"hubbard_square_U-{absU:.4f}_n{n:.3f}_L{L}_b{beta:.2f}-{sID}"
    )
    for pID in range(n_walkers):
        h5 = os.path.join(sim_dir, "bins", f"bins_pID-{pID}.h5")
        if not (os.path.isfile(h5) and os.path.getsize(h5) >= min_size_bytes):
            return False
    return True


def run_local(sID, absU, n, mu, L, beta, N_therm, N_measurements,
              N_bins, N_updates, n_walkers=1):
    """Run a single (|U|, β) point directly via mpiexecjl/mpirun (no SLURM)."""
    job_name  = f"hubbard_square_U{absU:.4f}_n{n:.3f}_L{L}_b{beta:.2f}-{sID}"
    jl_script = os.path.join(SIMULATIONS_DIR, f"{job_name}.jl")
    log_file  = os.path.join(SIMULATIONS_DIR, f"{job_name}.log")

    if _point_already_done(absU, n, L, beta, sID, n_walkers):
        print(f"[SKIP]  |U|={absU:.4f}  β={beta:.2f}  (HDF5 already populated)")
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SIMULATIONS_DIR, exist_ok=True)

    launcher = _resolve_launcher()
    cmd = [
        launcher, "-n", str(n_walkers),
        "julia", f"--project={REPO_ROOT}", jl_script,
        str(sID), str(absU), str(n), str(mu), str(L), str(beta),
        str(N_therm), str(N_measurements), str(N_bins), str(N_updates),
    ]
    print(f"\n[LOCAL] |U|={absU:.4f}  β={beta:.2f}  walkers={n_walkers}")
    print(f"        cmd: {' '.join(cmd)}")
    print(f"        log: {log_file}")
    with open(log_file, "w") as lf:
        rc = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=DATA_DIR,
        ).returncode
    if rc != 0:
        print(f"        FAILED (exit {rc}) — see {log_file}")
    else:
        print(f"        ok")
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="Submit / run the 2D attractive Hubbard β-sweep.",
    )
    parser.add_argument(
        "--abs-u", type=float, default=ABS_U,
        help=f"|U|/t for the run (Julia driver flips the sign). Default: {ABS_U}",
    )
    parser.add_argument(
        "--n", type=float, default=N_TARGET,
        help=f"Target filling ⟨n⟩. Default: {N_TARGET}",
    )
    args = parser.parse_args()

    BETA_GRIDS = {args.abs_u: np.linspace(2.0, 14.0, 25)}
    ABS_US = list(BETA_GRIDS.keys())

    sID = 1

    # Sampling parameters.  Hubbard local_updates! does one full HS-field sweep
    # per call.  100 sweeps between measurements (N_updates=100) ensures samples
    # are independent across the full β range.  1 measurement per bin → 100
    # bins/walker, 10,000 measurement sweeps + 2,000 thermalization sweeps per walker.
    N_bins         = N_BINS_PER_WALKER       # 100
    N_measurements = N_bins                  # 1 measurement per bin
    N_updates      = 100                     # sweeps per measurement
    N_therm        = 2_000                   # thermalization sweeps

    total_jobs    = sum(len(b) for b in BETA_GRIDS.values())
    total_samples = N_WALKERS * N_bins
    print(f"Parallel walkers : {N_WALKERS}")
    print(f"Bins per walker  : {N_bins}")
    print(f"Total samples    : {total_samples} per (|U|, β)")
    print(f"N_measurements   : {N_measurements}  N_updates: {N_updates}  N_therm: {N_therm}")
    print(f"Submitting {total_jobs} jobs across {len(ABS_US)} |U| values")
    print("-" * 60)

    print(f"Mode             : {'LOCAL (mpirun)' if LOCAL else 'SLURM (sbatch)'}")
    if LOCAL:
        print(f"Local concurrency: {LOCAL_PARALLEL} job(s) × {N_WALKERS} cores "
              f"= {LOCAL_PARALLEL * N_WALKERS} cores in use")

    # Build full job list first so we can run with controlled parallelism.
    job_list = []
    for absU in ABS_US:
        b_array = BETA_GRIDS[absU]
        print(
            f"\n|U|={absU:.4f}  ({len(b_array)} β points: "
            f"{b_array[0]:.2f} → {b_array[-1]:.2f})"
        )
        for beta in b_array:
            generate_jl_script(sID, absU, args.n, MU_INIT, L, beta,
                               N_therm, N_measurements, N_bins, N_updates)
            job_list.append((absU, beta))

    if not LOCAL:
        for absU, beta in job_list:
            generate_slurm_script(sID, absU, args.n, MU_INIT, L, beta,
                                  N_therm, N_measurements, N_bins, N_updates,
                                  n_walkers=N_WALKERS)
        return

    # ── LOCAL execution ────────────────────────────────────────────────────
    if LOCAL_PARALLEL <= 1:
        for absU, beta in job_list:
            run_local(sID, absU, args.n, MU_INIT, L, beta,
                      N_therm, N_measurements, N_bins, N_updates,
                      n_walkers=N_WALKERS)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=LOCAL_PARALLEL) as ex:
            futures = [
                ex.submit(run_local, sID, absU, args.n, MU_INIT, L, beta,
                          N_therm, N_measurements, N_bins, N_updates,
                          N_WALKERS)
                for absU, beta in job_list
            ]
            for f in futures:
                f.result()


if __name__ == "__main__":
    main()
