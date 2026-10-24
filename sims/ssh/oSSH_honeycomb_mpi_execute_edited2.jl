using LinearAlgebra
using DelimitedFiles
using Random
using Printf
using Statistics
using MPI
using JLD2
using DataFrames
using CSV

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner           as mt

# initialize MPI
MPI.Init()

function kvbs_phonon_config(L::Int, β::T, Δτ::T, R̄::T, ΔR::T, shift::Int = 0) where {T<:AbstractFloat}

    # Length of imaginary time axis
    Lτ = round(Int, β/Δτ)

    # Number of unit cells
    N = L^2

    # Number of sites in the lattice
    Nsites = N * 2

    # Number of phonon modes
    Nph = Nsites * 2

    # Initialize matrix to contain phonon fields
    x = zeros(T, Nph, Lτ) 

    # Get fields associated with each type of phonon field
    x′ = reshape(x, (L, L, 4, Lτ))
    Xa = @view x′[:, :, 1, :]
    Ya = @view x′[:, :, 2, :]
    Xb = @view x′[:, :, 3, :]
    Yb = @view x′[:, :, 4, :]

    # Iterate over unit cells in direction of second lattice vector
    for n2 in 1:L
        # Iterate over unit cells in direction of first lattice vector
        for n1 in 1:L
            # # sample displacement
            # rA = (R̄ + ΔR * randn())/2
            # rB = (R̄ + ΔR * randn())/2
            # Get unit cells label
            label = mod(n1 + 2*n2 + shift, 3)
            # shift phonon positions in unit cells according to the
            # three "types" of unit cell configurations that can occur
            if label == 0
                @. Xa[n1, n2, :] = +R̄ + ΔR*randn()/2
                @. Ya[n1, n2, :] = 0.0  + ΔR*randn()/2
                @. Xb[n1, n2, :] = -R̄ + ΔR*randn()/2
                @. Yb[n1, n2, :] = 0.0 + ΔR*randn()/2
            elseif label == 1
                @. Xa[n1, n2, :] = -R̄ * cos(π/3) + ΔR*randn()/2
                @. Ya[n1, n2, :] = +R̄ * sin(π/3) + ΔR*randn()/2
                @. Xb[n1, n2, :] = +R̄ * cos(π/3) + ΔR*randn()/2
                @. Yb[n1, n2, :] = +R̄ * sin(π/3) + ΔR*randn()/2
            else
                @. Xa[n1, n2, :] = -R̄ * cos(π/3) + ΔR*randn()/2
                @. Ya[n1, n2, :] = -R̄ * sin(π/3) + ΔR*randn()/2
                @. Xb[n1, n2, :] = +R̄ * cos(π/3) + ΔR*randn()/2
                @. Yb[n1, n2, :] = -R̄ * sin(π/3) + ΔR*randn()/2
            end
        end
    end

    return x
end

function eval_ρkvbs(x::AbstractArray{T}, unit_cell, lattice) where {T<:AbstractFloat}

    # Define the first nearest-neighbor bond in a honeycomb lattice.
    bond_1 = lu.Bond(orbitals = (1,2), displacement = [0,0])

    # Define the second nearest-neighbor bond in a honeycomb lattice.
    bond_2 = lu.Bond(orbitals = (1,2), displacement = [-1,0])

    # Define the third nearest-neighbor bond in a honeycomb lattice.
    bond_3 = lu.Bond(orbitals = (1,2), displacement = [0,-1])

    # number of phonon modes and length of imaginary time axis
    Nph, Lτ = size(x)

    # site per unit cell
    n = unit_cell.n

    # number of types of phonons
    nph = 4

    # numer of unit cells
    N = Nph ÷ nph

    # reshaped phonon fields for convience
    x′ = reshape(x, (N, nph, Lτ))

    # initialize order parameter to zero
    ρ_kvbs = zero(Complex{T})

    # K-point in momentum space
    Kp = [+2π/3, +2π/(3*sqrt(3))]
    Km = [+2π/3, -2π/(3*sqrt(3))]
    K  = Kp - Km

    # position space location of unit cell
    R = zeros(T, 2)

    # gives location of the unit cell
    loc = zeros(Int, 2)

    # iterate over unit cells
    for uA in 1:N

        # get site sublattice A site in unit cell
        iA = lu.loc_to_site(uA, 1, unit_cell)

        # get the location of the unit cell
        lu.unitcell_to_loc!(loc, uA, lattice)

        # get position of unit cell
        lu.loc_to_pos!(R, loc, unit_cell)

        # neighboring sublattice B site connected via bond 1
        jB1 = lu.site_to_site(iA, bond_1.displacement, 2, unit_cell, lattice)
        uB1 = lu.site_to_unitcell(jB1, unit_cell)

        # neighboring sublattice B site connected via bond 2
        jB2 = lu.site_to_site(iA, bond_2.displacement, 2, unit_cell, lattice)
        uB2 = lu.site_to_unitcell(jB2, unit_cell)

        # neighboring sublattice B site connected via bond 3
        jB3 = lu.site_to_site(iA, bond_3.displacement, 2, unit_cell, lattice)
        uB3 = lu.site_to_unitcell(jB3, unit_cell)

        # phonon positions associated with A site
        X_iA = @view x′[uA, 1, :]
        X̄_iA = mean(X_iA)
        Y_iA = @view x′[uA, 2, :]
        Ȳ_iA = mean(Y_iA)

        # calcualte phase factor
        expiKR = exp(-im*dot(K,R))

        # change in length of bond 1 contribuation to order parameter
        θ = 0.0
        X_jB1 = @view x′[uB1, 3, :]
        X̄_jB1 = mean(X_jB1)
        Y_jB1 = @view x′[uB1, 4, :]
        Ȳ_jB1 = mean(Y_jB1)
        ΔR_jB1 = cos(θ)*(X̄_jB1 - X̄_iA) + sin(θ)*(Ȳ_jB1 - Ȳ_iA)
        ρ_kvbs -= ΔR_jB1 * exp(im*θ) * expiKR

        # change in length of bond 2 contribuation to order parameter
        θ = -2π/3
        X_jB2 = @view x′[uB2, 3, :]
        X̄_jB2 = mean(X_jB2)
        Y_jB2 = @view x′[uB2, 4, :]
        Ȳ_jB2 = mean(Y_jB2)
        ΔR_jB2 = cos(θ)*(X̄_jB2 - X̄_iA) + sin(θ)*(Ȳ_jB2 - Ȳ_iA)
        ρ_kvbs -= ΔR_jB2 * exp(im*θ) * expiKR

        # change in length of bond 3 contribuation to order parameter
        θ = +2π/3
        X_jB3 = @view x′[uB3, 3, :]
        X̄_jB3 = mean(X_jB3)
        Y_jB3 = @view x′[uB3, 4, :]
        Ȳ_jB3 = mean(Y_jB3)
        ΔR_jB3 = cos(θ)*(X̄_jB3 - X̄_iA) + sin(θ)*(Ȳ_jB3 - Ȳ_iA)
        ρ_kvbs -= ΔR_jB3 * exp(im*θ) * expiKR
    end

    return ρ_kvbs/N
end

function eval_radial_displacement(x)

    # length of imaginary time axis
    Lτ = size(x, 2)

    # number of phonon modes
    Nph = size(x, 1)

    # number of unit cells
    N = Nph ÷ 4

    # number of sites
    Nsites = 2 * N

    # displacement for each site
    R  = zeros(eltype(x), Nsites)
    R′ = reshape(R, (2, N))

    # reshape phonon fields for convenience
    x′ = reshape(x, N, 4, Lτ)

    # iterate over the imaginary time axis
    for l in 1:Lτ
        # iterate over unit cells
        for n in 1:N
            # radial displacement for sublattice A site
            R′[1, n] += sqrt( x′[n, 1, l]^2 + x′[n, 2, l]^2 )
            # radial displacement for sublattice B site
            R′[2, n] += sqrt( x′[n, 3, l]^2 + x′[n, 4, l]^2 )
        end
    end

    # normalize the result
    @. R = R / Lτ

    return R
end

# Define top-level function for running the DQMC simulation.
function run_ossh_honeycomb_simulation(comm, sID, Ω, α, μ, β, L, N_burnin, N_updates, N_bins, ordered; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "ossh_honeycomb_w%.1f_a%.4f_m%.1f_L%d_b%.2f" Ω α μ L β

    # Get the MPI comm rank, which fixes the process ID (pID).
    pID = MPI.Comm_rank(comm)

    # Get the MPI com size
    N_pID = MPI.Comm_size(comm)

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = pID
    )

    # define checkpoint filename
    datafolder = simulation_info.datafolder
    sID = simulation_info.sID
    pID = simulation_info.pID

    checkpoint_name_old = @sprintf "checkpoint_sID%d_pID%d_old.jld2" sID pID
    checkpoint_name     = @sprintf "checkpoint_sID%d_pID%d.jld2" sID pID
    checkpoint_name_new = @sprintf "checkpoint_sID%d_pID%d_new.jld2" sID pID

    checkpoint_filename_old = joinpath(datafolder, checkpoint_name_old)
    checkpoint_filename     = joinpath(datafolder, checkpoint_name)
    checkpoint_filename_new = joinpath(datafolder, checkpoint_name_new)

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # Initialise site displacement directory if starting a new simulation
    site_displacement_folder = joinpath(simulation_info.datafolder, "site_displacement")
    site_displacement_file = joinpath(site_displacement_folder, "site_displacement_sID-$(sID)_pID-$(pID).txt")
    if !simulation_info.resuming
        mkpath(site_displacement_folder)
    end

    # Initialize directory for KVBS order parameter if starting a new simulation
    kvbs_order_parameter_folder = joinpath(simulation_info.datafolder, "kvbs_order_parameter")
    if !simulation_info.resuming
        mkpath(kvbs_order_parameter_folder)
    end

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # Initialize KVBS order parameter file for each process if starting a new simulation
    kvbs_order_parameter_file = joinpath(kvbs_order_parameter_folder, "kvbs_order_parameter_sID-$(sID)_pID-$(pID).txt")
    if !simulation_info.resuming
        open(kvbs_order_parameter_file, "w") do io
            write(io, "KVSB_ORDER_PARAMETER_REAL, KVBS_ORDER_PARAMETER_IMAG\n")
        end
    end

    # Initialize a random number generator that will be used throughout the simulation.
    seed = abs(rand(Int))
    rng = Xoshiro(seed)

    # Set the discretization in imaginary time for the DQMC simulation.
    Δτ = 0.05

    # This flag indicates whether or not to use the checkboard approximation to
    # represent the exponentiated hopping matrix exp(-Δτ⋅K)
    checkerboard = true

    # Whether the propagator matrices should be represented using the
    # symmetric form B = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    # or the asymetric form B = exp(-Δτ⋅V)⋅exp(-Δτ⋅K)
    symmetric = true

    # Set the initial period in imaginary time slices with which the Green's function matrices
    # will be recomputed using a numerically stable procedure.
    n_stab = 10

    # Specify the maximum allowed error in any element of the Green's function matrix that is
    # corrected by performing numerical stabiliziation.
    δG_max = 1e-6

    # Calculate the bin size.
    bin_size = div(N_updates, N_bins)

    # get the number of burnin bins (defined as number of updates per bin)
    N_bins_burnin = div(N_burnin, bin_size)

    # Number of fermionic time-steps in HMC update.
    Nt = 4

    # Fermionic time-step used in HMC update.
    Δt = (π/(2*Ω))/Nt

    # regularization
    reg = 0.0

    # starting point of thermalization udpates
    n_bin_burnin = 1

    # starting bin for measurement/simulation updates
    n_bin = 1

    # intialize errors associated with numerical instability to zero
    δG = zero(δG_max)
    δθ = zero(δG_max)

    # Initialize a dictionary to store additional information about the simulation.
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "hmc_acceptance_rate" => 0.0,
        "swap_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "dt" => Δt,
        "Nt" => Nt,
        "reg" => reg,
        "seed" => seed,
    )

    #######################
    ### DEFINE THE MODEL ##
    #######################

    # Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[3/2,√3/2],
                        [3/2,-√3/2]],
        basis_vecs   = [[0.,0.],  # orbital A 
                        [1.,0.]]) # orbital B

    # Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Get the number of sites in the lattice.
    N = lu.nsites(unit_cell, lattice)

    # Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define the first nearest-neighbor bond in a honeycomb lattice.
    bond_1 = lu.Bond(orbitals = (1,2), displacement = [0,0])

    # Add the first nearest-neighbor bond in a honeycomb lattice to the model.
    bond_1_id = add_bond!(model_geometry, bond_1)

    # Define the second nearest-neighbor bond in a honeycomb lattice.
    bond_2 = lu.Bond(orbitals = (1,2), displacement = [-1,0])

    # Add the second nearest-neighbor bond in a honeycomb lattice to the model.
    bond_2_id = add_bond!(model_geometry, bond_2)

    # Define the third nearest-neighbor bond in a honeycomb lattice.
    bond_3 = lu.Bond(orbitals = (1,2), displacement = [0,-1])

    # Add the third nearest-neighbor bond in a honeycomb lattice to the model.
    bond_3_id = add_bond!(model_geometry, bond_3)

    # define a negative version of each bond
    bond_n1 = lu.Bond(orbitals = (2,1), displacement = [0,0])
    bond_n1_id = add_bond!(model_geometry, bond_n1)
    bond_n2 = lu.Bond(orbitals = (1,2), displacement = [1,0])
    bond_n2_id = add_bond!(model_geometry, bond_n2)
    bond_n3 = lu.Bond(orbitals = (1,2), displacement = [0,1])
    bond_n3_id = add_bond!(model_geometry, bond_n3)

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the tight-binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_1,bond_2,bond_3], # defines hopping
        t_mean = [t,t,t],     # defines corresponding hopping amplitude
        μ = μ,            # set chemical potential
        ϵ_mean = [0.,0.]     # set the (mean) on-site energy
    )
    ###########################################################
    # e-ph relations
    # introduce ssh electron phonon coupling parameters
    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    # define phonons for each orbital in unit cell
    # Honeycomb lattice contain two orbitals
    # 1 ===> black ===> A 
    # 2 ===> blue  ===> B 
    # two phonons for each site in directions of x and y
    phonon_Ax = PhononMode(orbital = 1, Ω_mean = Ω);
    phonon_Ay = PhononMode(orbital = 1, Ω_mean = Ω);
    phonon_Bx = PhononMode(orbital = 2, Ω_mean = Ω);
    phonon_By = PhononMode(orbital = 2, Ω_mean = Ω);

    # add all phonon modes to the model
    phonon_Ax_id = add_phonon_mode!(electron_phonon_model = electron_phonon_model, phonon_mode = phonon_Ax)
    phonon_Ay_id = add_phonon_mode!(electron_phonon_model = electron_phonon_model, phonon_mode = phonon_Ay)
    phonon_Bx_id = add_phonon_mode!(electron_phonon_model = electron_phonon_model, phonon_mode = phonon_Bx)
    phonon_By_id = add_phonon_mode!(electron_phonon_model = electron_phonon_model, phonon_mode = phonon_By)
    
    # define optical SSH coupling
    # have three bonds 
    # bond_1 =====> lattice displacement=(0,0) ===> (AxBx,0)===>(1,0)α
    # bond_2 =====> lattice displacement=(-1,0)===> (AxBx,AyBy)===>(-1/2 , √3/2 )α
    # bond_3 =====> lattice displacement=(0,-1)===> (AxBx,AyBy)===>(-1/2 , -√3/2 )α
    
    ###########
    # Bond_1 ##
    ###########

    # in x direction non-zero α
    ssh_AxBx_B1 = SSHCoupling(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        phonon_modes = (phonon_Ax_id, phonon_Bx_id), # t_eff = t - α⋅(X_B - X_A)
        bond = bond_1,
        α_mean = α
    )
    ssh_AxBx_B1_id = add_ssh_coupling!(
        electron_phonon_model = electron_phonon_model,
        ssh_coupling = ssh_AxBx_B1,
        tight_binding_model = tight_binding_model
    )

    ############################################################
    ############################################################

    # in y direction zero α

    # zero coupling here


    ###########
    # Bond_2 ##
    ###########

    # in x direction non-zero α
    ssh_AxBx_B2 = SSHCoupling(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        phonon_modes = (phonon_Ax_id, phonon_Bx_id), # t_eff = t - α⋅(X_B - X_A)
        bond = bond_2,
        α_mean = -α*cos(π/3)
    )
    ssh_AxBx_B2_id = add_ssh_coupling!(
        electron_phonon_model = electron_phonon_model,
        ssh_coupling = ssh_AxBx_B2,
        tight_binding_model = tight_binding_model
    )

    ############################################################
    ############################################################

    # in y direction non-zero α
    ssh_AyBy_B2 = SSHCoupling(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        phonon_modes = (phonon_Ay_id, phonon_By_id), # t_eff = t - α⋅(X_B - X_A)
        bond = bond_2,
        α_mean = -α*sin(π/3)
    )
    ssh_AyBy_B2_id = add_ssh_coupling!(
        electron_phonon_model = electron_phonon_model,
        ssh_coupling = ssh_AyBy_B2,
        tight_binding_model = tight_binding_model
    )

    ###########
    # Bond_3 ##
    ###########

    # in x direction non-zero α
    ssh_AxBx_B3 = SSHCoupling(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        phonon_modes = (phonon_Ax_id, phonon_Bx_id), # t_eff = t - α⋅(X_B - X_A)
        bond = bond_3,
        α_mean = -α*cos(π/3)
    )
    ssh_AxBx_B3_id = add_ssh_coupling!(
        electron_phonon_model = electron_phonon_model,
        ssh_coupling = ssh_AxBx_B3,
        tight_binding_model = tight_binding_model
    )

    ############################################################
    ############################################################

    # in y direction non-zero α
    ssh_AyBy_B3 = SSHCoupling(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        phonon_modes = (phonon_Ay_id, phonon_By_id), # t_eff = t - α⋅(X_B - X_A)
        bond = bond_3,
        α_mean = α*sin(π/3)
    )
    ssh_AyBy_B3_id = add_ssh_coupling!(
        electron_phonon_model = electron_phonon_model,
        ssh_coupling = ssh_AyBy_B3,
        tight_binding_model = tight_binding_model
    )


    ####################################
    ## BRANCHING CHECKPOINT BEHAVIOR: ##
    ## START OR RESUME SIMULATION     ##
    ####################################

    # if resuming a previous simulation
    if simulation_info.resuming

        # boolean to indicate whether a checkpoint was successfully loaded
        loaded_checkpoint = false

        # initialize checkpoint to nothing
        checkpoint = nothing

        # read in new checkpoint
        if isfile(checkpoint_filename_new)
            try
                # load new checkpoint
                checkpoint = JLD2.load(checkpoint_filename_new)
                # record that the checkpoint was successfully loaded
                loaded_checkpoint = true
            catch
                nothing
            end
        end

        # read in current checkpoint if necessary
        if isfile(checkpoint_filename) && !loaded_checkpoint
            try
                # load current checkpoint
                checkpoint = JLD2.load(checkpoint_filename)
                # record that the checkpoint was successfully loaded
                loaded_checkpoint = true
            catch
                nothing
            end
        end

        # read in old checkpoint if necessary
        if isfile(checkpoint_filename_old) && !loaded_checkpoint
            try
                # load old checkpoint
                checkpoint = JLD2.load(checkpoint_filename_old)
                # record that the checkpoint was successfully loaded
                loaded_checkpoint = true
            catch
                nothing
            end
        end

        # throw error if checkpoint not successfully loaded
        if !loaded_checkpoint
            error("Failed to load checkpoint successfully!")
        end
        
        # unpack the checkpoint, which is admittedly a little tedious
        rng                        = checkpoint["rng"]
        additional_info            = checkpoint["additional_info"]
        n_bin_burnin               = checkpoint["n_bin_burnin"]
        n_bin                      = checkpoint["n_bin"]
        measurement_container      = checkpoint["measurement_container"]
        tight_binding_parameters   = checkpoint["tight_binding_parameters"]
        electron_phonon_parameters = checkpoint["electron_phonon_parameters"]
        δG                         = checkpoint["dG"]
        δθ                         = checkpoint["dtheta"]
        n_stab                     = checkpoint["n_stab"]

    else
    
        ####################################
        ## Write a model summary to file. ##
        ####################################

        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
        )

        #################################################
        ### INITIALIZE FINITE LATTICE MODEL PARAMETERS ##
        #################################################

        # Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize electron-phonon parameters.
        electron_phonon_parameters = ElectronPhononParameters(
            β = β, Δτ = Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = rng
        )

        # if starting from an ordered configuration
        if ordered

            # characterisitic length scale of phonon mode
            X0 = 1/sqrt(2*Ω)

            # initialize ordered phonon config
            if iszero(mod(N_pID, 3))
                x_ordered = kvbs_phonon_config(L, β, Δτ, 3*X0, 1*X0, mod(pID,3))
            else
                x_ordered = kvbs_phonon_config(L, β, Δτ, 3*X0, 1*X0, rand(rng,0:2))
            end

            # record ordered phonon config
            copyto!(electron_phonon_parameters.x, x_ordered)
        end

        ##############################
        ### INITIALIZE MEASUREMENTS ##
        ##############################

        # Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        # Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        # Initialize the electron-phonon interaction related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        # Initialize the single-particle electron Green's function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "greens",
        #     time_displaced = true,
        #     pairs = [
        #         (1, 1), (2, 2), (1, 2)
        #     ]
        # )
        
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = false,
            pairs = [
                (1, 1), (2, 2), (1, 2)
            ]
        )

        # Initialize density correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "density",
        #     time_displaced = false,
        #     integrated = true,
        #     pairs = [
        #         (1, 1), (2, 2), (1, 2)
        #     ]
        # )

        # # Initialize density correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "density",
        #     time_displaced = false,
        #     integrated = true,
        #     pairs = [
        #         (1, 1)
        #     ]
        # )

        # Initialize the pair correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "pair",
        #     time_displaced = false,
        #     integrated = true,
        #     pairs = [
        #         (1, 1), (2, 2), (1, 2),
        #         (bond_1_id, bond_1_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id),
        #         (bond_2_id, bond_1_id), (bond_2_id, bond_2_id), (bond_2_id, bond_3_id),
        #         (bond_3_id, bond_1_id), (bond_3_id, bond_2_id), (bond_3_id, bond_3_id)
        #     ]
        # )


        # Initialize the spin-z correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "spin_z",
        #     time_displaced = false,
        #     integrated = true,
        #     pairs = [(1, 1), (2, 2), (1, 2)]
        # )

        # Initialize the bond correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "bond",
            time_displaced = false,
            integrated = true,
            pairs = [
                (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id),
                (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id)
            ]
        )


        # Initialize the sub-directories to which the various measurements will be written.
        initialize_measurement_directories(
            simulation_info = simulation_info,
            measurement_container = measurement_container
        )

        # write initial checkpoint
        JLD2.jldsave(
            checkpoint_filename;
            rng, additional_info,
            bin_size,
            N_burnin,
            N_bins_burnin, n_bin_burnin,
            N_updates,
            N_bins, n_bin,
            measurement_container,
            tight_binding_parameters,
            electron_phonon_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
        )

        # Added part!
        mkpath(joinpath(simulation_info.datafolder,"Phonon_positions"))

        # Synchronize all the MPI processes.
        # We need to ensure the sub-directories the measurements will be written are created
        # prior to letting any of the processes move beyond this point.
        MPI.Barrier(comm)

    end

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################


    # Allocate fermion path integral type.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the fermion path integral type with respect to electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Allocate and initialize propagators for each imaginary time slice.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize fermion greens calculator.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    # Initialize alternate fermion greens calculator required for performing various global updates.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    # Allocate equal-time Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize equal-time Green's function matrix
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt, reg = reg
    )

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # perform thermalization/burnin updates
    for bin in n_bin_burnin:N_bins_burnin

        # iterate of bin size
        for n in 1:bin_size

            # Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

            # perform swap update
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng,
                phonon_type_pairs = (
                    (phonon_Ax_id, phonon_Ax_id),
                    (phonon_Ay_id, phonon_Ay_id),
                    (phonon_Bx_id, phonon_Bx_id),
                    (phonon_By_id, phonon_By_id)
                )
            )

            # Record whether the swap update was accepted or rejected.
            additional_info["swap_acceptance_rate"] += accepted

        end

        # write new checkpoint
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            bin_size, N_burnin,
            N_bins_burnin, n_bin_burnin = bin + 1,
            N_updates,
            N_bins, n_bin,
            measurement_container,
            tight_binding_parameters, electron_phonon_parameters,
            dG = δG, dtheta = δθ, n_stab = fermion_greens_calculator.n_stab
        )
        # make current checkpoint the old checkpoint
        mv(checkpoint_filename, checkpoint_filename_old, force = true)
        # make new checkpoint the current checkpoint
        mv(checkpoint_filename_new, checkpoint_filename, force = true)
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Re-initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in n_bin:N_bins

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            # Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

            # perform swap update
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng,
                phonon_type_pairs = (
                    (phonon_Ax_id, phonon_Ax_id),
                    (phonon_Ay_id, phonon_Ay_id),
                    (phonon_Bx_id, phonon_Bx_id),
                    (phonon_By_id, phonon_By_id)
                )
            )

            # Record whether the swap update was accepted or rejected.
            additional_info["swap_acceptance_rate"] += accepted

            # Make measurements.
            (logdetG, sgndetG, δG, δθ) = make_measurements!(
                measurement_container,
                logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (electron_phonon_parameters,)
            )

            # Measure KVBS order parameter and write it to file (every update, high statistics)
            x = electron_phonon_parameters.x
            ρ_kvsb = eval_ρkvbs(x, unit_cell, lattice)
            open(kvbs_order_parameter_file, "a") do io
                @printf io "%.10f, %.10f\n" real(ρ_kvsb) imag(ρ_kvsb)
            end
        end

        # ── Per-bin phonon field snapshot (tau=0) ─────────────────────────────────
        # Written ONCE per bin (not per update) so the file has exactly N_bins rows.
        # Format: one CSV row with (L*L*nph + 1) values = 576 phonon positions + β.
        # Memory layout of the flat row: [Xa[1..N], Ya[1..N], Xb[1..N], Yb[1..N], β]
        # where N = L*L unit cells, nph = 4 modes (Xa, Ya = sublattice A; Xb, Yb = B).
        # In Python: reshape row[:-1] → (nph=4, N=144) or (nph=4, L, L) in F-order.
        let
            nph_loc  = 4
            Lτ_loc   = round(Int, β / Δτ)
            x_now    = electron_phonon_parameters.x          # (L*L*nph, Lτ)
            # tau=0 slice: first column of the (L*L*nph × Lτ) matrix
            xp       = reshape(x_now, (L * L * nph_loc, Lτ_loc))[:, 1]  # (576,)
            ph_file  = joinpath(simulation_info.datafolder, "Phonon_positions",
                                "phonon_config_sID$(sID)_pID$(pID).csv")
            open(ph_file, "a") do io
                # Write 576 phonon values followed by β on one CSV row
                for v in xp
                    @printf io "%.10f," v
                end
                @printf io "%.10f\n" β
            end
        end

        # ── Per-bin site radial displacement (τ-averaged, ML-ready) ───────────────
        # R[i] = mean over imaginary time of sqrt(Xx_i^2 + Xy_i^2) per site.
        # Shape: 2*L*L = 288 (sublattice A sites, then sublattice B sites).
        # Written once per bin; format: 288 values + β on one CSV row.
        # This is domain-invariant (only magnitudes) and gives clean PCA clusters.
        let
            x_now   = electron_phonon_parameters.x
            R_now   = eval_radial_displacement(x_now)        # (2*L*L,) = (288,)
            rd_file = joinpath(simulation_info.datafolder, "Phonon_positions",
                               "radial_displacement_sID$(sID)_pID$(pID).csv")
            open(rd_file, "a") do io
                for v in R_now
                    @printf io "%.10f," v
                end
                @printf io "%.10f\n" β
            end
        end

        # Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )

        # write new checkpoint
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            bin_size, N_burnin, 
            N_bins_burnin, n_bin_burnin = N_bins_burnin + 1,
            N_updates, N_bins, n_bin = bin + 1,
            measurement_container,
            tight_binding_parameters, electron_phonon_parameters,
            dG = δG, dtheta = δθ, n_stab = fermion_greens_calculator.n_stab
        )
        # make the current checkpoint the old checkpoint
        mv(checkpoint_filename, checkpoint_filename_old, force = true)
        # make new checkpoint the current checkpoint
        mv(checkpoint_filename_new, checkpoint_filename, force = true)
    end

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    # Synchronize all the MPI processes.
    # Before we prcoess the binned data to get the final averages and error bars
    # we need to make sure all the simulations running in parallel have run to
    # completion.
    MPI.Barrier(comm)

    # Have the primary MPI process calculate the final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(comm, simulation_info.datafolder, N_bins)

    # ordering K vector grid point
    n1 = 1 * L ÷ 3
    n2 = 2 * L ÷ 3

    # KVBS structure factor
    S_kvbs_avg, S_kvbs_err = composite_correlation_stat(comm;
        folder = simulation_info.datafolder,
        correlations = [
            "bond", "bond", "bond",
            "bond", "bond", "bond"
        ],
        spaces = [
            "momentum", "momentum", "momentum",
            "momentum", "momentum", "momentum"
        ],
        types = [
            "equal-time", "equal-time", "equal-time",
            "equal-time", "equal-time", "equal-time"
        ],
        ids = [
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id),
            (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id)
        ],
        locs = [
            (n1, n2), (n1, n2), (n1, n2),
            (n1, n2), (n1, n2), (n1, n2)
        ],
        Δls = [
            0, 0, 0,
            0, 0, 0
        ],
        num_bins = N_bins,
        f = (S11, S22, S33, S12, S13, S23) -> (S11 + S22 + S33 - real(S12) - real(S13) - real(S23))
    )
    additional_info["S_kvbs_avg"] = real(S_kvbs_avg)
    additional_info["S_kvbs_err"] = S_kvbs_err


    # KVBS susceptibility
    Chi_kvbs_avg, Chi_kvbs_err = composite_correlation_stat(comm;
        folder = simulation_info.datafolder,
        correlations = [
            "bond", "bond", "bond",
            "bond", "bond", "bond"
        ],
        spaces = [
            "momentum", "momentum", "momentum",
            "momentum", "momentum", "momentum"
        ],
        types = [
            "integrated", "integrated", "integrated",
            "integrated", "integrated", "integrated"
        ],
        ids = [
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id),
            (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id)
        ],
        locs = [
            (n1, n2), (n1, n2), (n1, n2),
            (n1, n2), (n1, n2), (n1, n2)
        ],
        Δls = [
            0, 0, 0,
            0, 0, 0
        ],
        num_bins = N_bins,
        f = (S11, S22, S33, S12, S13, S23) -> (S11 + S22 + S33 - real(S12) - real(S13) - real(S23))
    )
    additional_info["Chi_kvbs_avg"] = real(Chi_kvbs_avg)
    additional_info["Chi_kvbs_err"] = Chi_kvbs_err


    # KVBS correlation ratio
    R_kvbs_avg, R_kvbs_err = composite_correlation_stat(comm;
        folder = simulation_info.datafolder,
        correlations = [
            "bond", "bond", "bond", "bond", "bond", "bond",
            "bond", "bond", "bond", "bond", "bond", "bond",
            "bond", "bond", "bond", "bond", "bond", "bond",
            "bond", "bond", "bond", "bond", "bond", "bond",
            "bond", "bond", "bond", "bond", "bond", "bond"
        ],
        spaces = [
            "momentum", "momentum", "momentum", "momentum", "momentum", "momentum",
            "momentum", "momentum", "momentum", "momentum", "momentum", "momentum",
            "momentum", "momentum", "momentum", "momentum", "momentum", "momentum",
            "momentum", "momentum", "momentum", "momentum", "momentum", "momentum",
            "momentum", "momentum", "momentum", "momentum", "momentum", "momentum"
        ],
        types = [
            "equal-time", "equal-time", "equal-time", "equal-time", "equal-time", "equal-time",
            "equal-time", "equal-time", "equal-time", "equal-time", "equal-time", "equal-time",
            "equal-time", "equal-time", "equal-time", "equal-time", "equal-time", "equal-time",
            "equal-time", "equal-time", "equal-time", "equal-time", "equal-time", "equal-time",
            "equal-time", "equal-time", "equal-time", "equal-time", "equal-time", "equal-time"
        ],
        ids = [
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id),
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id),
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id),
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id),
            (bond_1_id, bond_1_id), (bond_2_id, bond_2_id), (bond_3_id, bond_3_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id), (bond_2_id, bond_3_id)
        ],
        locs = [
            (n1,   n2), (n1,   n2), (n1,   n2), (n1,   n2), (n1,   n2), (n1,   n2),
            (n1+1, n2), (n1+1, n2), (n1+1, n2), (n1+1, n2), (n1+1, n2), (n1+1, n2),
            (n1-1, n2), (n1-1, n2), (n1-1, n2), (n1-1, n2), (n1-1, n2), (n1-1, n2),
            (n1, n2+1), (n1, n2+1), (n1, n2+1), (n1, n2+1), (n1, n2+1), (n1, n2+1),
            (n1, n2-1), (n1, n2-1), (n1, n2-1), (n1, n2-1), (n1, n2-1), (n1, n2-1)
        ],
        Δls = [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        num_bins = N_bins,
        f = (
            S11,    S22,    S33,    S12,    S13,    S23,
            S11pq1, S22pq1, S33pq1, S12pq1, S13pq1, S23pq1,
            S11mq1, S22mq1, S33mq1, S12mq1, S13mq1, S23mq1,
            S11pq2, S22pq2, S33pq2, S12pq2, S13pq2, S23pq2,
            S11mq2, S22mq2, S33mq2, S12mq2, S13mq2, S23mq2,
        ) -> (
              S11pq1 + S22pq1 + S33pq1 - real(S12pq1) - real(S13pq1) - real(S23pq1)
            + S11mq1 + S22mq1 + S33mq1 - real(S12mq1) - real(S13mq1) - real(S23mq1)
            + S11pq2 + S22pq2 + S33pq2 - real(S12pq2) - real(S13pq2) - real(S23pq2)
            + S11mq2 + S22mq2 + S33mq2 - real(S12mq2) - real(S13mq2) - real(S23mq2)
        )/(
              S11 + S22 + S33 - real(S12) - real(S13) - real(S23)
        )/4
    )
    additional_info["R_kvbs_avg"] = real(R_kvbs_avg)
    additional_info["R_kvbs_err"] = R_kvbs_err


    # measure extended s-wave pair susceptibility
    # P_ext_s_avg, P_ext_s_err = composite_correlation_stat(comm;
    #     folder = simulation_info.datafolder,
    #     correlations = [
    #         "pair", "pair", "pair",
    #         "pair", "pair", "pair",
    #         "pair", "pair", "pair"
    #     ],
    #     spaces = [
    #         "momentum", "momentum", "momentum",
    #         "momentum", "momentum", "momentum",
    #         "momentum", "momentum", "momentum"
    #     ],
    #     types = [
    #         "integrated", "integrated", "integrated",
    #         "integrated", "integrated", "integrated",
    #         "integrated", "integrated", "integrated"
    #     ],
    #     ids = [
    #         (bond_1_id, bond_1_id), (bond_1_id, bond_2_id), (bond_1_id, bond_3_id),
    #         (bond_2_id, bond_2_id), (bond_2_id, bond_2_id), (bond_2_id, bond_3_id),
    #         (bond_3_id, bond_1_id), (bond_3_id, bond_2_id), (bond_3_id, bond_3_id)
    #     ],
    #     locs = [
    #         (0, 0), (0, 0), (0, 0),
    #         (0, 0), (0, 0), (0, 0),
    #         (0, 0), (0, 0), (0, 0)
    #     ],
    #     Δls = [
    #         0, 0, 0,
    #         0, 0, 0,
    #         0, 0, 0
    #     ],
    #     num_bins = N_bins,
    #     f = (
    #         P11,  P12,  P13,  P21,  P22,  P23,  P31,  P32,  P33
    #     ) -> (
    #         P11 + P12 + P13 + P21 + P22 + P23 + P31 + P32 + P33
    #     )/3
    # )
    # additional_info["P_ext_s_avg"] = real(P_ext_s_avg)
    # additional_info["P_ext_s_err"] = P_ext_s_err

    # Calculate acceptance rates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["swap_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    return nothing
end