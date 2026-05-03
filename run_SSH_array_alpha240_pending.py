"""
Submit the pending alpha=0.240 jobs (beta 17.5 → 25.0) as a single SLURM
job array instead of 16 individual jobs.

Strategy
--------
* One static Julia runner script  (oSSH_array_runner.jl)  – reused by every
  array task; all parameters are passed as CLI arguments.
* One SLURM array script         (run_alpha240_pending.sh)  – uses
  $SLURM_ARRAY_TASK_ID to pick the right beta from a bash array.

Usage
-----
    python run_SSH_array_alpha240_pending.py          # generate + submit
    python run_SSH_array_alpha240_pending.py --dry-run  # generate only
"""

import subprocess
import sys
import numpy as np

# ── Source file ───────────────────────────────────────────────────────────────
SOURCE_FILE = "sims/ssh/oSSH_honeycomb_mpi_execute_edited2.jl"

# ── Simulation parameters ─────────────────────────────────────────────────────
SID       = 5
W         = 0.5
ALPHA     = 0.240
MU        = 0.0
L         = 12
ORDERED   = 0

N_WALKERS         = 4
N_BINS_TOTAL      = 1000
N_BINS_PER_WALKER = N_BINS_TOTAL // N_WALKERS   # 250
N_BINS            = N_BINS_PER_WALKER
N_UPDATES         = N_BINS * 100                # bin_size = 100 → 25,000
N_BURNIN          = 10_000

# ── Pending beta grid for alpha=0.240 ─────────────────────────────────────────
# Full grid is np.linspace(5, 30, 51) with 0.5 spacing.
# Already done: beta < 17.5.  Remaining: 17.5, 18.0, …, 25.0
BETAS = list(np.arange(17.5, 25.5, 0.5))   # 17.5 → 25.0, step 0.5  (16 values)

JULIA_RUNNER  = "oSSH_array_runner.jl"
SLURM_SCRIPT  = "run_alpha240_pending.sh"
JOB_NAME      = f"oSSH_a{ALPHA:.4f}_array"


# ── 1. Generic Julia runner ───────────────────────────────────────────────────
def write_julia_runner():
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
    n_tasks = len(BETAS)
    # Build a bash array literal:  ( "17.50" "18.00" … )
    beta_list = " ".join(f'"{b:.2f}"' for b in BETAS)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition=puma,puma-i9",
        f"#SBATCH -J {JOB_NAME}",
        # %A = array master job ID, %a = task index
        f"#SBATCH -o {JOB_NAME}-%A_%a.output",
        f"#SBATCH --ntasks={N_WALKERS}",
        f"#SBATCH --cpus-per-task=1",
        f"#SBATCH --array=0-{n_tasks - 1}",
        "",
        "# Beta values indexed by SLURM_ARRAY_TASK_ID",
        f"BETAS=( {beta_list} )",
        "BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}",
        "",
        f"time mpirun -n {N_WALKERS} julia {JULIA_RUNNER} \\",
        f"    {SID} {W} {ALPHA} {MU} ${{BETA}} {L} \\",
        f"    {N_BURNIN} {N_UPDATES} {N_BINS} {ORDERED}",
    ]

    with open(SLURM_SCRIPT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SLURM_SCRIPT}  ({n_tasks} tasks, array 0-{n_tasks - 1})")
    print(f"  beta range : {BETAS[0]:.2f} → {BETAS[-1]:.2f}")
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
