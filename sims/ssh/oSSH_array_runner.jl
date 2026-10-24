using MPI
include("oSSH_honeycomb_mpi_execute_edited2.jl")

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
