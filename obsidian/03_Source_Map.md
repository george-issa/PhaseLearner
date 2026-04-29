# 03 — Source Map

External references that the project leans on. Cite by section anchor when adding a claim to [[04_Claims_and_Proof_Obligations]].

## FSS reference β_c (SSH oSSH, w=0.5)

Source: `sohan_phase_diagram.pdf` (in repo root or shared with the user); finite-size scaling on bond-bond order parameter for the optical SSH model on L = 12 honeycomb at ω = 0.5, m = 0.0.

| α      | β_c   |
|--------|-------|
| 0.240  | 17.5  |
| 0.245  | 14.3  |
| 0.250  | 12.5  |
| 0.255  | 12.3  |

Status for 0.235 / 0.2415 / 0.2435: TBD until 1000-bin runs land.

## Bond Holstein paper (FSS β_c)

`BondHolstein_paper.pdf` (in repo root). Coupling convention: `λ_user = α²/2`; paper's `λ_bond = 2 · λ_user`. Fig. 7's x-axis is `1/λ_user`; our α set maps to x = 5, 4, 3, 2, 1.

| α       | 1/λ   | T_c  | β_c   | Quality |
|---------|-------|------|-------|---------|
| 0.6325  | 5.0   | 0.63 | 1.58  | explicit anchor |
| 0.7071  | 4.0   | 0.85 | 1.18  | interpolated |
| 0.8165  | 3.0   | 1.07 | 0.93  | near-anchor |
| 1.0000  | 2.0   | 1.5  | 0.67  | extrapolated |
| 1.4142  | 1.0   | 2.0  | 0.50  | left edge of Fig. 7 |

## Attractive Hubbard T_c reference

Paiva, dos Santos, Scalettar, Denteneer, *Critical temperature for the two-dimensional attractive Hubbard model*, PRB 69, 184501 (2004). At U = −4, ⟨n⟩ = 0.875: T_c ≈ 0.118 t → β_c ≈ 8.5.

## SmoQyDQMC.jl

The Julia DQMC package powering all simulations. Tutorial we modelled the attractive-Hubbard driver on: `tutorials/hubbard_square_density_tuning.jl`. Local install on the user's Tower lives at `~/Hubbard_2d/Project.toml` + `Manifest.toml` (juliaup, JLL-bundled MPICH backend). MPI launcher: `~/.julia/bin/mpiexecjl`.

## Project files on cluster

(See also [[05_Code_Map#cluster-side scripts]].)

- `oSSH_honeycomb_mpi_execute_edited2.jl` — Julia DQMC driver for SSH honeycomb (MPI, 4 walkers × 250 bins).
- `run_SSH_edited2.py` / `run_SSH_array_alpha240_pending.py` — SLURM submission, per-α β grid (see `SIMULATION_UPDATES.txt` §2).
- `convert_jld2_csv_edited.jl` — extracts BondBond and Green's-function CSVs from JLD2 (loops over all α).
- `extract_phonons_edited.py` — stacks per-walker phonon CSVs into ML-ready files (`phonon_tau0_*` + `radial_disp_*`).
- `attractive_hubbard_square_mpi_execute.jl` — Hubbard pilot DQMC driver.
- `run_attractive_hubbard.py` — local (mpirun) / SLURM (sbatch) Hubbard runner.
- `create_csv_pair_betasweep.py` — Hubbard pair-momentum extractor that reads HDF5 directly.

## Cluster-environment cheatsheet

- **HPC SLURM cluster**: SLURM submission (`run_SSH_edited2.py`).
- **`ls-260108-phy` Dell Tower** (24 cores, no SLURM): home on NFS at `/nfs/home/gissa/`. Auto-activated `~/Hubbard_2d/<venv-dir>/` + `JULIA_PROJECT=$HOME/Hubbard_2d`. Use `mpiexecjl` (NOT system `mpirun`). MPI backend is JLL-bundled MPICH — single-machine only; for multi-node switch to `MPIPreferences.use_system_binary()` and re-precompile.

## Per-α β grids (SSH oSSH)

From `SIMULATION_UPDATES.txt` §2. All grids use 0.5 spacing.

| α       | β range  | # points |
|---------|----------|----------|
| 0.235   | [5, 23]  | 37       |
| 0.240   | [5, 30]  | 51 (extended; β_c ≈ 17.5) |
| 0.2415  | [5, 27]  | 45 (interpolated; β_c ≈ 16) |
| 0.2435  | [5, 25]  | 41 (interpolated; β_c ≈ 15) |
| 0.245   | [5, 23]  | 37       |
| 0.250   | [4, 20]  | 33       |
| 0.255   | [4, 20]  | 33       |
| **Total** |        | **277 SLURM jobs** |
