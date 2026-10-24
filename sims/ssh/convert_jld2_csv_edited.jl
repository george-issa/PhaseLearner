using SmoQyDQMC
using Dates
using Printf

###################################################
PATH_MAIN = "/nfs/home/gissa/SSH/Data"
sID = @sprintf("%d", 1)
w   = @sprintf("%.1f", 0.5)
L   = 12

# All alpha values — must match run_SSH_edited.py BETA_GRIDS keys
ALPHAS = [0.235, 0.240, 0.2415, 0.2435, 0.245, 0.250, 0.255]
###################################################

# Observables to extract.  Each entry is (correlation, type, space).
# correlation_bins_to_csv(-1; ...) uses N_bins=-1 meaning "all available bins".
# With N_WALKERS parallel walkers each running N_BINS_PER_WALKER bins,
# the function automatically combines all pID subdirectories → total rows
# per output CSV = N_WALKERS × N_BINS_PER_WALKER (e.g. 4 × 250 = 1000).
OBSERVABLES = [
    ("bond",   "equal-time", "position"),
    ("bond",   "equal-time", "momentum"),
    ("greens", "equal-time", "momentum"),
    ("greens", "equal-time", "position"),
]

for a_val in ALPHAS

    a = @sprintf("%.4f", a_val)
    println()
    println(repeat("=", 20) * " alpha = $a " * repeat("=", 20))

    # Collect all simulation folders for this alpha (all beta values)
    entries = readdir(PATH_MAIN)
    folders = filter(entry -> isdir(joinpath(PATH_MAIN, entry)), entries)

    DQMC_DIR = filter(name -> startswith(name, "ossh_honeycomb_w$(w)") &&
                              contains(name, "a$(a)")                   &&
                              contains(name, "L$(L)")                   &&
                              endswith(name, "-$(sID)"), folders)
    PATH_DIR = [joinpath(PATH_MAIN, DIR_NAME) for DIR_NAME in DQMC_DIR]

    if isempty(PATH_DIR)
        println("  [skip] no folders found for alpha=$(a)")
        continue
    end

    println("  Found $(length(PATH_DIR)) beta folders")

    for (CORR, TYPE, SPACE) in OBSERVABLES

        println()
        println(repeat("-", 10) *
                " Correlation: $CORR | Type: $TYPE | Space: $SPACE " *
                repeat("-", 10))

        for PATH_FL in PATH_DIR
            println("  Processing: $PATH_FL")
            correlation_bins_to_csv(-1;
                folder          = PATH_FL,
                correlation     = CORR,
                type            = TYPE,
                space           = SPACE,
                write_index_key = true
            )
        end
    end
end

println()
println(repeat("=", 60))
println("Done. Run create_bondbond / create_greens to build ML-ready CSVs.")
println(repeat("=", 60))
