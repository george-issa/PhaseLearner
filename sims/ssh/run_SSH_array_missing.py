"""
Submit the 7 missing (alpha, beta) points of the sID=5 1000-bin sweep as a
single SLURM job array.

Missing points (audit of data/ssh/sim_data, 2026-06-10):
    alpha=0.240 : beta = 25.00                  (job died ~Mar 9; checkpoints
                                                 exist, driver auto-resumes)
    alpha=0.245 : beta = 14.00, 22.50, 23.00    (never ran)
    alpha=0.255 : beta = 15.00, 15.50, 16.00    (never ran)

Parameters match the 192 completed runs exactly (verified uniform across all
simulation_info_pID*_sID5.toml files): sID=5, w=0.5, mu=0.0, L=12,
4 walkers x 250 bins (bin_size=100), N_burnin=10000, ordered=0.

All generated files (oSSH_array_runner.jl, run_missing_sweep.sh, job .output)
are written next to this script, under sims/ssh/.

IMPORTANT: the Julia driver creates its data folder in the directory sbatch
is invoked from (filepath="."), so run this script from the SAME directory
the original sweep was submitted from — the existing
ossh_honeycomb_w0.5_a0.2400_m0.0_L12_b25.00-5 folder must be there for the
beta=25.00 checkpoint resume to pick it up.

Usage
-----
    python sims/ssh/run_SSH_array_missing.py            # generate + submit
    python sims/ssh/run_SSH_array_missing.py --dry-run  # generate only
"""

import os
import subprocess
import sys

# ── Paths (everything lives next to this script, under sims/ssh/) ─────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = "oSSH_honeycomb_mpi_execute_edited2.jl"   # same dir as the runner

# ── Simulation parameters (identical to the 192 completed sID=5 runs) ─────────
SID       = 5
W         = 0.5
MU        = 0.0
L         = 12
ORDERED   = 0

N_WALKERS         = 4
N_BINS_TOTAL      = 1000
N_BINS_PER_WALKER = N_BINS_TOTAL // N_WALKERS   # 250
N_BINS            = N_BINS_PER_WALKER
N_UPDATES         = N_BINS * 100                # bin_size = 100 → 25,000
N_BURNIN          = 10_000

# ── Missing (alpha, beta) points ──────────────────────────────────────────────
POINTS = [
    (0.240, 25.00),   # checkpointed crash — resumes from checkpoint_sID5_pID{0..3}.jld2
    (0.245, 14.00),
    (0.245, 22.50),
    (0.245, 23.00),
    (0.255, 15.00),
    (0.255, 15.50),
    (0.255, 16.00),
]

JULIA_RUNNER  = os.path.join(SCRIPT_DIR, "oSSH_array_runner.jl")
SLURM_SCRIPT  = os.path.join(SCRIPT_DIR, "run_missing_sweep.sh")
JOB_NAME      = "oSSH_missing_array"


# ── 1. Generic Julia runner ───────────────────────────────────────────────────
def write_julia_runner():
    # include() resolves relative to the runner's own location (sims/ssh),
    # so the runner works no matter where julia is launched from.
    content = f"""\
using MPI
include("{SOURCE_FILE}")

MPI.Init()
comm = MPI.COMM_WORLD

sID      = parse(Int,     ARGS[1])
w        = parse(Float64, ARGS[2])
a        = parse(Float64, ARGS[3])
m        = parse(Float64, ARGS[4])
b        = parse(Float64, ARGS[5])
L        = parse(Int,     ARGS[6])
N_burnin = parse(Int,     ARGS[7])
N_updates= parse(Int,     ARGS[8])
N_bins   = parse(Int,     ARGS[9])
ordered  = parse(Bool,    ARGS[10])

run_ossh_honeycomb_simulation(comm, sID, w, a, m, b, L,
                               N_burnin, N_updates, N_bins, ordered)
MPI.Finalize()
"""
    with open(JULIA_RUNNER, "w") as f:
        f.write(content)
    print(f"Wrote {JULIA_RUNNER}")


# ── 2. SLURM array script ─────────────────────────────────────────────────────
def write_slurm_array():
    n_tasks = len(POINTS)
    alpha_list = " ".join(f'"{a:.4f}"' for a, _ in POINTS)
    beta_list  = " ".join(f'"{b:.2f}"' for _, b in POINTS)

    lines = [
        "#!/bin/bash",
        "#SBATCH --partition=puma,puma-i9",
        f"#SBATCH -J {JOB_NAME}",
        # %A = array master job ID, %a = task index
        f"#SBATCH -o {os.path.join(SCRIPT_DIR, JOB_NAME)}-%A_%a.output",
        f"#SBATCH --ntasks={N_WALKERS}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --array=0-{n_tasks - 1}",
        "",
        "# (alpha, beta) pairs indexed by SLURM_ARRAY_TASK_ID",
        f"ALPHAS=( {alpha_list} )",
        f"BETAS=( {beta_list} )",
        "ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}",
        "BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}",
        "",
        f"time mpirun -n {N_WALKERS} julia {JULIA_RUNNER} \\",
        f"    {SID} {W} ${{ALPHA}} {MU} ${{BETA}} {L} \\",
        f"    {N_BURNIN} {N_UPDATES} {N_BINS} {ORDERED}",
    ]

    with open(SLURM_SCRIPT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SLURM_SCRIPT}  ({n_tasks} tasks, array 0-{n_tasks - 1})")
    for i, (a, b) in enumerate(POINTS):
        print(f"  task {i}: alpha={a:.4f}  beta={b:.2f}")
    print(f"  walkers/task: {N_WALKERS}  |  bins/walker: {N_BINS}  |  "
          f"total samples: {N_WALKERS * N_BINS}")


# ── 3. Submit ─────────────────────────────────────────────────────────────────
def submit():
    result = subprocess.run(["sbatch", SLURM_SCRIPT], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Submitted: {result.stdout.strip()}")
    else:
        print(f"sbatch failed:\n{result.stderr}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    write_julia_runner()
    write_slurm_array()

    if dry_run:
        print("\nDry-run mode: scripts written but NOT submitted.")
    else:
        submit()
