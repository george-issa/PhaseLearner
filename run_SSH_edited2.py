import numpy as np
import subprocess

source_file = "sims/oSSH_honeycomb_mpi_execute_edited2.jl"

# ── Parallel walkers ───────────────────────────────────────────────────────────
# N_WALKERS independent Markov chains run inside each SLURM job via MPI.
# Each walker contributes N_BINS_PER_WALKER measurements; the total sample
# count per (alpha, beta) point is N_WALKERS × N_BINS_PER_WALKER.
#
# Speed trade-off:
#   N_WALKERS=1, N_BINS=1000  →  1000 samples, baseline wall-clock
#   N_WALKERS=4, N_BINS=250   →  1000 samples, ~4× faster wall-clock  ← recommended
#   N_WALKERS=4, N_BINS=1000  →  4000 samples, same wall-clock (more stats)
#
# Rule: set N_BINS_PER_WALKER = N_BINS_TOTAL // N_WALKERS
N_WALKERS       = 4
N_BINS_TOTAL    = 1000
N_BINS_PER_WALKER = N_BINS_TOTAL // N_WALKERS   # 250

# ── Per-alpha beta grids ───────────────────────────────────────────────────────
# Grids are chosen so the deep-ordered phase is sampled well above beta_c.
# All grids use 0.5 spacing (approx).  Adjust as needed.
#
#   alpha=0.235 : no known beta_c in accessible range → sample broadly
#   alpha=0.240 : beta_c≈17.5  → extend to 30 so ordered phase is well sampled
#   alpha=0.2415: beta_c≈16    → extend to 27
#   alpha=0.2435: beta_c≈15    → extend to 25
#   alpha=0.245 : beta_c≈14.3  → extend to 23
#   alpha=0.250 : beta_c≈12.5  → up to 20
#   alpha=0.255 : beta_c≈12.3  → up to 20
#
BETA_GRIDS = {
    # 0.235:  np.linspace(5, 30, 51),   # 0.5 spacing, deep-disordered anchor
    0.240:  np.linspace(5, 30, 51),   # extended: need deep ordered (beta_c=17.5)
    # 0.2415: np.linspace(5, 27, 45),   # interpolated between 0.240 and 0.2435
    0.2425: np.linspace(5, 27, 45),
    # 0.2435: np.linspace(5, 25, 41),   # interpolated between 0.2415 and 0.245
    0.245:  np.linspace(5, 23, 37),   # beta_c=14.3
    0.250:  np.linspace(4, 20, 33),   # beta_c=12.5
    0.255:  np.linspace(4, 20, 33),   # beta_c=12.3
}

# All alphas to run (in order)
ALPHAS = list(BETA_GRIDS.keys())


def generate_jl_script(sID, w, a, m, b, L, N_burnin, N_updates, N_bins, ordered):

    script_content = f"""
using MPI
include("{source_file}")

# Initialize the MPI communicator
MPI.Init()
comm = MPI.COMM_WORLD

# Read in the command line arguments.
sID = parse(Int, ARGS[1])   # simulation ID
w = parse(Float64, ARGS[2])
a = parse(Float64, ARGS[3])
m = parse(Float64, ARGS[4]) # initial chemical potential
b = parse(Float64, ARGS[5])
L = parse(Int, ARGS[6])
N_burnin = parse(Int, ARGS[7])
N_updates = parse(Int, ARGS[8])
N_bins = parse(Int, ARGS[9])
ordered = parse(Bool, ARGS[10])

# Run the simulation
run_ossh_honeycomb_simulation(comm, sID, w, a, m, b, L, N_burnin, N_updates, N_bins, ordered)

# Finalize the MPI communicator
MPI.Finalize()
    """

    script_name = f"oSSH_honeycomb_L{L}_w{w:.1f}_a{a:.4f}_m{m:.1f}_b{b:.2f}_L{L}-{sID}.jl"
    with open(script_name, 'w') as script_file:
        script_file.write(script_content)


def generate_slurm_script(sID, w, a, m, b, L, N_burnin, N_updates, N_bins, ordered,
                          n_walkers=1):

    job_name = f"oSSH_honeycomb_L{L}_w{w:.1f}_a{a:.4f}_m{m:.1f}_b{b:.2f}_L{L}-{sID}"
    jl_script = f"{job_name}.jl"
    script_name = f"{job_name}.sh"

    with open(script_name, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --partition=puma,puma-i9\n")
        f.write(f"#SBATCH -J {job_name}\n")
        f.write(f"#SBATCH -o {job_name}-%j.output\n")
        # One MPI task per walker; each task is single-threaded.
        f.write(f"#SBATCH --ntasks={n_walkers}\n")
        f.write(f"#SBATCH --cpus-per-task=1\n")
        # set a cap time if needed
        # f.write(f"#SBATCH --time=72:00:00\n\n")
        # MPI is installed system-wide (/usr/bin/mpirun) — no module load needed.

        # srun launches one MPI task per --ntasks slot automatically.
        # Each task gets a unique MPI rank (pID 0 … n_walkers-1).
        f.write(f"time mpirun -n {n_walkers} julia {jl_script} "
                f"{sID} {w} {a} {m} {b} {L} "
                f"{N_burnin} {N_updates} {N_bins} {ordered}\n")

    # Submit Slurm job
    subprocess.run(['sbatch', script_name])
    print(f'Submitted: a={a:.4f}  b={b:.2f}  walkers={n_walkers}  '
          f'bins/walker={N_bins}  total_bins={n_walkers * N_bins}')


def main():

    # Simulation ID
    sID = 5

    # Phonon frequency
    w = 0.5

    # Chemical potential
    m = 0.0

    # Lattice size
    L = 12

    # ── Sampling parameters ────────────────────────────────────────────────────
    # bin_size = N_updates_per_walker / N_bins_per_walker = 25,000 / 250 = 100.
    # bin_size stays at 100 (same autocorrelation safety as original runs).
    # Total samples per (alpha, beta) = N_WALKERS × N_BINS_PER_WALKER = 1000.
    # Wall-clock time ≈ single-walker run with 250 bins → ~4× faster.
    N_bins    = N_BINS_PER_WALKER          # 250 per walker
    N_updates = N_bins * 100              # bin_size = 100  →  25,000
    N_burnin  = 10_000                    # thermalization (same regardless of N_bins)

    # Ordered or disordered initial state (0 = disordered)
    ordered = 0

    bin_size = N_updates // N_bins
    total_jobs = sum(len(betas) for betas in BETA_GRIDS.values())
    total_samples = N_WALKERS * N_bins
    print(f"Parallel walkers : {N_WALKERS}")
    print(f"Bins per walker  : {N_bins}  (bin_size={bin_size})")
    print(f"Total samples    : {total_samples} per (alpha, beta)")
    print(f"N_updates/walker : {N_updates}  |  N_burnin: {N_burnin}")
    print(f"Submitting {total_jobs} jobs across {len(ALPHAS)} alpha values")
    print("-" * 60)

    for a in ALPHAS:
        b_array = BETA_GRIDS[a]
        print(f"\nalpha={a:.4f}  ({len(b_array)} beta points: "
              f"{b_array[0]:.2f} → {b_array[-1]:.2f})")
        for b in b_array:
            generate_jl_script(sID, w, a, m, b, L, N_burnin, N_updates, N_bins, ordered)
            generate_slurm_script(sID, w, a, m, b, L, N_burnin, N_updates, N_bins,
                                  ordered, n_walkers=N_WALKERS)


if __name__ == "__main__":
    main()
