#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Local environment setup for PhaseLearner (DQMC + ML pipeline).
#
# Sets up two things:
#   1. Conda env "phaselearner" from environment.yml  (Python ML side)
#   2. Julia project in this repo with MPI + SmoQyDQMC, plus mpiexecjl on PATH
#                                                       (DQMC simulation side)
#
# Tested on macOS (Apple Silicon) and Linux x86_64.
# Run from the repo root:  bash setup_local.sh
# ----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=============================================================="
echo "PhaseLearner local setup"
echo "Repo root: $REPO_ROOT"
echo "=============================================================="

# ── 1. Python (conda) ────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
    echo
    echo "[1/2] Creating conda env 'phaselearner' from environment.yml ..."
    if conda env list | awk '{print $1}' | grep -qx "phaselearner"; then
        echo "      env already exists — updating instead"
        conda env update -n phaselearner -f environment.yml --prune
    else
        conda env create -f environment.yml
    fi
    echo "      activate later with:  conda activate phaselearner"
else
    echo
    echo "[1/2] SKIP: conda not found on PATH."
    echo "      Install Miniforge (recommended on Apple Silicon):"
    echo "        https://github.com/conda-forge/miniforge"
    echo "      Then re-run this script, or create the env manually:"
    echo "        conda env create -f environment.yml"
fi

# ── 2. Julia (project + MPI + SmoQyDQMC) ────────────────────────────────────
if ! command -v julia >/dev/null 2>&1; then
    echo
    echo "[2/2] ERROR: julia not found on PATH."
    echo "      Install via juliaup:"
    echo "        curl -fsSL https://install.julialang.org | sh"
    echo "      Then re-run this script."
    exit 1
fi

echo
echo "[2/2] Setting up Julia project in $REPO_ROOT ..."
julia --project=. -e '
using Pkg
Pkg.add(["MPI", "MPIPreferences", "SmoQyDQMC"])
using MPIPreferences
MPIPreferences.use_jll_binary()      # JLL-bundled MPICH; no system MPI needed
Pkg.instantiate()
Pkg.precompile()
'

# Also install MPI in the *global* Julia env so the standalone mpiexecjl
# launcher (which doesn't pass --project=.) can resolve `using MPI` from
# any working directory.
echo
echo "      Adding MPI to the global Julia env (so mpiexecjl works anywhere) ..."
julia -e '
using Pkg
Pkg.add(["MPI", "MPIPreferences"])
using MPIPreferences
MPIPreferences.use_jll_binary()
'

# Install mpiexecjl into ~/.julia/bin
JULIA_BIN="$HOME/.julia/bin"
mkdir -p "$JULIA_BIN"
julia --project=. -e "using MPI; MPI.install_mpiexecjl(destdir=\"$JULIA_BIN\", force=true)"

# PATH hint (don't auto-edit shell rc; just tell the user)
case ":$PATH:" in
    *":$JULIA_BIN:"*)
        ;;
    *)
        echo
        echo "NOTE: $JULIA_BIN is not on your PATH."
        echo "      Add this to ~/.zshrc (or ~/.bashrc):"
        echo "        export PATH=\"\$HOME/.julia/bin:\$PATH\""
        ;;
esac

# ── Done ────────────────────────────────────────────────────────────────────
cat <<'DONE'

==============================================================
Setup complete.

Quick checks:
  conda activate phaselearner
  python -c "import torch, umap, sklearn; print('python ok')"
  mpiexecjl --version
  julia --project=. -e 'using SmoQyDQMC, MPI; println("julia ok")'

Before launching DQMC runs, pin BLAS to 1 thread to avoid
oversubscription when MPI walkers run in parallel:

  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1

Then launch a sweep:
  python run_attractive_hubbard.py
==============================================================
DONE
