#!/usr/bin/env bash
# Validate the LBC pipeline (lbc_train.py) on the 2D Ising model end to end.
# Generates 4x4 snapshots, runs the LBC sweep, and plots the W-curve whose
# central peak should land on the exact Tc = 2/ln(1+sqrt2) = 2.269 J.
#
# Usage:  ./run_ising_check.sh [output_dir]
# Override the interpreter with:  PYTHON=/path/to/python ./run_ising_check.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/phaselearner/bin/python}"
OUT="${1:-$ROOT/results/lbc_ising_check}"
WORK="$OUT/work"                     # bulky regenerable intermediates (gitignored)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$WORK"

echo "[1/3] Generating 4x4 Ising snapshots ..."
"$PY" "$HERE/make_ising_csv.py" "$WORK/ising_L4.csv" \
    --L 4 --tmin 1.0 --tmax 3.6 --ntemp 27 --chains 300 --warmup 2000

echo "[2/3] Running LBC sweep (27 CNNs) ..."
"$PY" "$ROOT/scripts/lbc_train.py" "$WORK/ising_L4.csv" "$WORK/lbc_out" \
    --tc-min 1.0 --tc-max 3.6 --num-tc 27 \
    --L 4 --batch-size 64 --num-epochs 80 --patience-lr 6 --patience-stop 12

echo "[3/3] Plotting W-curve ..."
"$PY" "$HERE/plot_lbc.py" "$WORK/lbc_out" "$OUT/lbc_ising_L4_Wcurve.png"
cp "$WORK"/lbc_out/accs_*.csv "$OUT/ising_L4_accs.csv"
cp "$WORK"/lbc_out/tc_grid_*.csv "$OUT/ising_L4_tcgrid.csv"

echo
echo "Done. Figure: $OUT/lbc_ising_L4_Wcurve.png"
