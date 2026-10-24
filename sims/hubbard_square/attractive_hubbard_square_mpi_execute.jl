# 2D attractive square-lattice Hubbard model — MPI driver
# =========================================================
# Mirrors the SmoQyDQMC suite tutorial `hubbard_square_density_tuning.jl`
# (https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/tutorials/hubbard_square_density_tuning.jl)
# with three adjustments:
#   1. Sign-fixed positive |U| input on the command line; the driver flips the
#      sign internally so SLURM submitter sees the same convention as the rest
#      of the PhaseLearner pipeline.
#   2. Folder name uses `_U%.4f_n%.3f_L%d_b%.2f` so the Python extractor can
#      glob output directories deterministically.
#   3. Per-walker bin files are left in their canonical SmoQyDQMC layout
#      (datafolder/equal-time/pair/momentum/) — the Python extractor reads
#      those bins directly and reshapes them into ML-ready CSVs.
#
# Filling: ⟨n⟩=0.875 by dynamic MuTuner (Phys. Rev. E 105, 045311).
# Sign problem: attractive Hubbard is sign-problem-free at any filling on a
# bipartite lattice; MuTuner is set with `complex_sign_problem = false`.
#
# Measurements that PhaseLearner consumes:
#   * pair correlation (1,1)  →  on-site s-wave SC structure factor P_s(q) — PRIMARY
#   * greens (1,1) [time-displaced] → gap signal, secondary G(k) observable
#   * density (1,1) integrated → CDW competition cross-check

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework  as dqmcf
import SmoQyDQMC.MuTuner          as mt

using Random
using Printf
using MPI


function run_attractive_hubbard_square_simulation(
    comm::MPI.Comm;
    sID,            # simulation ID
    U,              # Hubbard interaction (NEGATIVE for attractive)
    n,              # target filling ⟨n⟩
    μ,              # initial chemical potential (will be tuned by MuTuner)
    L,              # linear lattice size
    β,              # inverse temperature
    N_therm,        # thermalization sweeps
    N_measurements, # number of measurement events
    N_bins,         # number of bin-averaged outputs (≤ N_measurements)
    N_updates,      # local-update sweeps between measurements
    Δτ            = 0.05,
    n_stab        = 10,
    δG_max        = 1e-6,
    symmetric     = false,
    checkerboard  = false,
    seed          = abs(rand(Int)),
    filepath      = ".",
)

    # ── Folder & MPI setup ────────────────────────────────────────────────
    datafolder_prefix = @sprintf "hubbard_square_U%.4f_n%.3f_L%d_b%.2f" U n L β

    pID = MPI.Comm_rank(comm)

    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = (L > 10),
        sID = sID,
        pID = pID,
    )
    initialize_datafolder(comm, simulation_info)

    # ── Metadata ──────────────────────────────────────────────────────────
    rng = Xoshiro(seed)
    metadata = Dict{String,Any}(
        "U"                          => U,
        "n_target"                   => n,
        "mu_init"                    => μ,
        "N_therm"                    => N_therm,
        "N_measurements"             => N_measurements,
        "N_bins"                     => N_bins,
        "N_updates"                  => N_updates,
        "n_stab_init"                => n_stab,
        "dG_max"                     => δG_max,
        "symmetric"                  => symmetric,
        "checkerboard"               => checkerboard,
        "seed"                       => seed,
        "local_acceptance_rate"      => 0.0,
        "reflection_acceptance_rate" => 0.0,
    )

    # ── Lattice geometry ──────────────────────────────────────────────────
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs   = [[0.0, 0.0]],
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Bonds: ±x, ±y nearest neighbors (no t′ for the attractive Hubbard pilot)
    bond_px = lu.Bond(orbitals = (1,1), displacement = [ 1, 0]); bond_px_id = add_bond!(model_geometry, bond_px)
    bond_py = lu.Bond(orbitals = (1,1), displacement = [ 0, 1]); bond_py_id = add_bond!(model_geometry, bond_py)
    bond_nx = lu.Bond(orbitals = (1,1), displacement = [-1, 0]); bond_nx_id = add_bond!(model_geometry, bond_nx)
    bond_ny = lu.Bond(orbitals = (1,1), displacement = [ 0,-1]); bond_ny_id = add_bond!(model_geometry, bond_ny)

    # ── Tight-binding & Hubbard model ────────────────────────────────────
    t = 1.0
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py],
        t_mean  = [t,        t       ],
        t_std   = [0.0,      0.0     ],
        ϵ_mean  = [0.0],
        ϵ_std   = [0.0],
        μ       = μ,
    )

    hubbard_model = HubbardModel(
        ph_sym_form = true,
        U_orbital   = [1],
        U_mean      = [U],   # NEGATIVE → attractive
        U_std       = [0.0],
    )

    model_summary(
        simulation_info     = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry      = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions        = (hubbard_model,),
    )

    # ── Parameters ────────────────────────────────────────────────────────
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry      = model_geometry,
        rng                 = rng,
    )
    hubbard_params = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model  = hubbard_model,
        rng            = rng,
    )
    hst_parameters = HubbardSpinHirschHST(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_params,
        rng                = rng,
    )

    # ── MuTuner: dynamic μ adjustment toward target ⟨n⟩ ───────────────────
    # NB: `complex_sign_problem` selects the logger's element type (ComplexF64
    # vs Float64), not whether a physical sign problem exists.  Attractive U
    # has no physical sign problem, but the imaginary HS field makes the
    # Green's functions complex — so the logger must be complex too.
    chemical_potential_tuner = mt.init_mutunerlogger(
        target_density             = n,
        inverse_temperature        = β,
        system_size                = lu.nsites(unit_cell, lattice),
        initial_chemical_potential = μ,
        complex_sign_problem       = (U < 0),
    )

    # ── Measurements ──────────────────────────────────────────────────────
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)
    initialize_measurements!(measurement_container, tight_binding_model)
    initialize_measurements!(measurement_container, hubbard_model)

    # G(k,τ) — gap signal (time-displaced for full Matsubara structure)
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "greens",
        time_displaced        = true,
        pairs                 = [(1, 1)],
    )

    # Density-density (CDW competition cross-check)
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "density",
        time_displaced        = false,
        integrated            = true,
        pairs                 = [(1, 1)],
    )

    # **PRIMARY OBSERVABLE for ML**: on-site s-wave pair-pair → P_s(q).
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "pair",
        time_displaced        = false,
        integrated            = true,
        pairs                 = [(1, 1)],
    )

    # Spin-z (zero by symmetry for attractive Hubbard, but sanity check)
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "spin_z",
        time_displaced        = false,
        integrated            = true,
        pairs                 = [(1, 1)],
    )

    # ── DQMC simulation setup ─────────────────────────────────────────────
    # Attractive U → complex HST → Green's functions must be complex.
    fermion_path_integral_up = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic   = false,
    )
    fermion_path_integral_dn = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic   = false,
    )
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_params)
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

    Bup = initialize_propagators(fermion_path_integral_up, symmetric = symmetric, checkerboard = checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric = symmetric, checkerboard = checkerboard)

    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    Gup_ττ = similar(Gup); Gup_τ0 = similar(Gup); Gup_0τ = similar(Gup)
    Gdn_ττ = similar(Gdn); Gdn_τ0 = similar(Gdn); Gdn_0τ = similar(Gdn)

    δG = zero(logdetGup); δθ = zero(logdetGup)

    # ── Thermalization (with MuTuner) ─────────────────────────────────────
    for _ in 1:N_therm
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn) = update_chemical_potential!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn;
            chemical_potential_tuner     = chemical_potential_tuner,
            tight_binding_parameters     = tight_binding_parameters,
            fermion_path_integral_up     = fermion_path_integral_up,
            fermion_path_integral_dn     = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn,
        )
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up     = fermion_greens_calculator_up,
            fermion_greens_calculator_dn     = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng,
        )
        metadata["reflection_acceptance_rate"] += accepted

        (acc, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn,
            δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = true,
        )
        metadata["local_acceptance_rate"] += acc
    end

    # Reset stability diagnostics for measurement phase
    δG = zero(logdetGup); δθ = zero(logdetGup)
    bin_size = N_measurements ÷ N_bins

    # ── Measurement loop ──────────────────────────────────────────────────
    for measurement in 1:N_measurements
        for _ in 1:N_updates
            (logdetGup, sgndetGup, logdetGdn, sgndetGdn) = update_chemical_potential!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn;
                chemical_potential_tuner     = chemical_potential_tuner,
                tight_binding_parameters     = tight_binding_parameters,
                fermion_path_integral_up     = fermion_path_integral_up,
                fermion_path_integral_dn     = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn,
            )
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up     = fermion_greens_calculator_up,
                fermion_greens_calculator_dn     = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng,
            )
            metadata["reflection_acceptance_rate"] += accepted

            (acc, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn,
                δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = true,
            )
            metadata["local_acceptance_rate"] += acc
        end

        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
            measurement_container,
            logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
            logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn,
            δG_max = δG_max, δG = δG, δθ = δθ,
            model_geometry           = model_geometry,
            tight_binding_parameters = tight_binding_parameters,
            coupling_parameters      = (hubbard_params, hst_parameters),
        )

        write_measurements!(
            measurement_container = measurement_container,
            simulation_info       = simulation_info,
            model_geometry        = model_geometry,
            measurement           = measurement,
            bin_size              = bin_size,
            Δτ                    = Δτ,
        )
    end

    # ── Finalize ─────────────────────────────────────────────────────────
    merge_bins(simulation_info)

    metadata["local_acceptance_rate"]      /= (N_therm + N_measurements * N_updates)
    metadata["reflection_acceptance_rate"] /= (N_therm + N_measurements * N_updates)
    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab
    metadata["dG"] = δG

    save_simulation_info(simulation_info, metadata)

    save_density_tuning_profile(
        simulation_info, chemical_potential_tuner;
        export_to_h5  = true,
        export_to_csv = false,
    )

    process_measurements(
        comm;
        datafolder         = simulation_info.datafolder,
        n_bins             = N_bins,
        export_to_csv      = true,
        scientific_notation = false,
        decimals           = 7,
        delimiter          = " ",
    )

    # s-wave pair correlation ratio at q=(0,0).
    # SmoQyDQMC indexes the BZ with 0-indexed offsets (the AFM tutorial uses
    # `(L÷2, L÷2)` for q=(π,π), which only matches the 0-indexed convention).
    # Γ point → (0,0); the four nearest k-shell neighbors (with periodic BZ).
    # Wrapped in try/catch so a diagnostic-only failure does not erase the
    # measurement bins that already succeeded.
    try
        Rsc, ΔRsc = compute_correlation_ratio(
            comm;
            datafolder           = simulation_info.datafolder,
            correlation          = "pair",
            type                 = "equal-time",
            id_pairs             = [(1, 1)],
            id_pair_coefficients = [1.0],
            q_point              = (0, 0),
            q_neighbors          = [(1, 0), (L-1, 0), (0, 1), (0, L-1)],
        )
        metadata["Rsc_mean_real"] = real(Rsc)
        metadata["Rsc_mean_imag"] = imag(Rsc)
        metadata["Rsc_std"]       = ΔRsc
        save_simulation_info(simulation_info, metadata)
    catch err
        @warn "compute_correlation_ratio (s-wave SC) failed; skipping diagnostic" exception = (err, catch_backtrace())
    end

    return nothing
end


# ── Script entry point ────────────────────────────────────────────────────
# Positional ARGS so the SLURM submitter can call with:
#   srun julia attractive_hubbard_square_mpi_execute.jl \
#        sID  |U|  n  μ_init  L  β  N_therm  N_measurements  N_bins  N_updates
# The driver flips the sign internally so |U| is passed as a positive number.

if abspath(PROGRAM_FILE) == @__FILE__
    MPI.Init()
    comm = MPI.COMM_WORLD

    sID            = parse(Int,     ARGS[1])
    absU           = parse(Float64, ARGS[2])     # PASS |U| (positive)
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
        U              = -absU,   # attractive
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
end
