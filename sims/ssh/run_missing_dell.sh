#!/usr/bin/env bash
# ============================================================================
# run_missing_dell.sh — fully self-contained driver for the 7 missing
# sID=5 oSSH simulations on a 24-core Dell Tower. No arguments needed:
#
#     cd ~/Documents/SSH/PhaseLearner/sims/ssh
#     nohup bash run_missing_dell.sh > driver.log 2>&1 &
#
# What it does, in order:
#   1. Sanity checks (julia 1.8.5, oSSH_env, mpiexecjl, data folder).
#   2. Launches the beta=14.00 job as a CANARY and waits (up to 60 min)
#      for its first checkpoint — proof the whole stack works end to end.
#   3. Launches the remaining 6 jobs (7 jobs x 4 walkers = 28 ranks on
#      24 cores, ~17% oversubscription — intended).
#   4. Waits for everything and prints a completion summary.
#
# Missing points (audit of the 1000-bin sweep, 2026-06-10):
#   alpha=0.240 : beta = 25.00                (resumes: 74/100 burn-in bins done)
#   alpha=0.245 : beta = 14.00, 22.50, 23.00
#   alpha=0.255 : beta = 15.00, 15.50, 16.00
# Parameters mirror the 192 completed runs: 4 walkers x 250 bins
# (bin_size=100), N_burnin=10000, ordered=0, sID=5.
#
# Safety / restart: completed jobs (global_stats.csv present) are skipped;
# jobs already running are detected and NOT relaunched; partial jobs resume
# from their last checkpoint. => Re-running this script after a crash or
# reboot is always safe — just run the same command again.
#
# Stop everything:  pkill -f oSSH_array_runner.jl
# Logs: driver narration on stdout; one .output file per job next to script.
# ============================================================================

set -u
unset LD_LIBRARY_PATH    # shell-level hygiene; ranks also get env -u (below)

# ── Configuration (baked in; env vars can override if ever needed) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/nfs/home/gissa/Documents/SSH/PhaseLearner/data/ssh/sim_data}"
PROJ="${PROJ:-$HOME/oSSH_env}"                 # julia 1.8.5 + SmoQyDQMC 1.0.17
JULIA="${JULIA:-julia +1.8.5}"                 # juliaup channel pin
MPIEXECJL="${MPIEXECJL:-$HOME/.julia/bin/mpiexecjl}"
N_WALKERS=4

# Simulation parameters (must match the completed sID=5 runs)
SID=5; W=0.5; MU=0.0; L=12
N_BURNIN=10000; N_UPDATES=25000; N_BINS=250; ORDERED=0

CANARY="0.2450:14.00"        # shortest job; first checkpoint ~25 min in
CANARY_TIMEOUT_MIN=60
REST=(
  "0.2400:25.00"             # resumes from existing checkpoints
  "0.2450:23.00"
  "0.2450:22.50"
  "0.2550:16.00"
  "0.2550:15.50"
  "0.2550:15.00"
)
ALL_JOBS=("$CANARY" "${REST[@]}")

START_EPOCH=$(date +%s)

say() { echo "[$(date '+%F %T')] $*"; }

folder_of() {   # folder_of alpha beta
  echo "ossh_honeycomb_w${W}_a${1}_m${MU}_L${L}_b${2}-${SID}"
}

is_running() {  # is_running alpha beta — matches the rank command line
  pgrep -f "oSSH_array_runner.jl ${SID} ${W} ${1} ${MU} ${2} " >/dev/null 2>&1
}

# launch_job alpha beta — sets JOB_STATE (complete|running|launched) and LAUNCH_PID
launch_job() {
  local A="$1" B="$2" folder log
  folder=$(folder_of "$A" "$B")
  log="$SCRIPT_DIR/oSSH_missing_a${A}_b${B}.output"
  JOB_STATE=""; LAUNCH_PID=0

  if [ -f "$folder/global_stats.csv" ]; then
    say "[skip  ] a=$A b=$B — already complete"
    JOB_STATE="complete"; return 0
  fi
  if is_running "$A" "$B"; then
    say "[runng] a=$A b=$B — already running, not relaunching"
    JOB_STATE="running"; return 0
  fi
  if [ -f "$folder/checkpoint_sID${SID}_pID0.jld2" ]; then
    say "[resume] a=$A b=$B — continuing from checkpoint"
  else
    say "[start ] a=$A b=$B"
  fi
  # env -u strips the LD_LIBRARY_PATH that mpiexecjl's host julia (juliaup
  # DEFAULT channel, 1.12.x) injects — without it the 1.8.5 ranks load
  # mismatched SuiteSparse/OpenBLAS libs. $JULIA unquoted on purpose: it
  # carries the +1.8.5 channel flag.
  "$MPIEXECJL" --project="$PROJ" -n "$N_WALKERS" \
      env -u LD_LIBRARY_PATH $JULIA --project="$PROJ" "$SCRIPT_DIR/oSSH_array_runner.jl" \
      "$SID" "$W" "$A" "$MU" "$B" "$L" \
      "$N_BURNIN" "$N_UPDATES" "$N_BINS" "$ORDERED" \
      > "$log" 2>&1 &
  LAUNCH_PID=$!
  JOB_STATE="launched"
  say "         pid $LAUNCH_PID  log: $log"
}

# ── 1. Sanity checks ─────────────────────────────────────────────────────────
say "host: $(hostname) | data: $DATA_DIR | project: $PROJ"

[ -x "$MPIEXECJL" ] || { say "ERROR: mpiexecjl not found at $MPIEXECJL"; exit 1; }
[ -d "$PROJ" ]      || { say "ERROR: Julia project not found at $PROJ"; exit 1; }
$JULIA --version >/dev/null 2>&1 || { say "ERROR: '$JULIA' not runnable (is ~/.juliaup/bin on PATH?)"; exit 1; }
case "$($JULIA --version)" in
  *1.8.5*) : ;;
  *) say "ERROR: '$JULIA' is not julia 1.8.5"; exit 1 ;;
esac
cd "$DATA_DIR" || { say "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }

b25_folder=$(folder_of 0.2400 25.00)
if [ ! -f "$b25_folder/checkpoint_sID${SID}_pID0.jld2" ] && [ ! -f "$b25_folder/global_stats.csv" ]; then
  say "WARNING: $b25_folder has no checkpoints — beta=25.00 would restart from scratch (~33 h extra)."
fi

# ── 2. Canary: beta=14.00 ────────────────────────────────────────────────────
A="${CANARY%%:*}"; B="${CANARY##*:}"
launch_job "$A" "$B"

if [ "$JOB_STATE" != "complete" ]; then
  # Gate on the ROTATED checkpoint (_old.jld2): the driver writes an initial
  # checkpoint immediately at startup, so the plain checkpoint file proves
  # only that init worked. The _old file appears when the SECOND checkpoint
  # is written, i.e. after one full bin of real updates (~20 min at b=14).
  cfile="$(folder_of "$A" "$B")/checkpoint_sID${SID}_pID0_old.jld2"
  # healthy = rotated checkpoint younger than 80 min before script start
  # (covers both a fresh rotation after launch and a live already-running job)
  threshold=$((START_EPOCH - 4800))
  say "[canary] waiting for first checkpoint (expect ~25 min; timeout ${CANARY_TIMEOUT_MIN} min)..."
  mins=0
  while :; do
    mt=$(stat -c %Y "$cfile" 2>/dev/null || echo 0)
    if [ "$mt" -ge "$threshold" ]; then
      say "[canary] healthy — checkpoint detected. Launching the remaining jobs."
      break
    fi
    if [ "$LAUNCH_PID" -gt 0 ] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      say "ERROR: canary exited before its first checkpoint — check $SCRIPT_DIR/oSSH_missing_a${A}_b${B}.output"
      say "Remaining jobs NOT launched. Fix the problem, then rerun this script."
      exit 1
    fi
    if [ "$mins" -ge "$CANARY_TIMEOUT_MIN" ]; then
      say "ERROR: no canary checkpoint after ${CANARY_TIMEOUT_MIN} min. Remaining jobs NOT launched."
      say "Check the canary log; if it is healthy but slow, simply rerun this script — it will detect the running canary and continue."
      exit 1
    fi
    sleep 60; mins=$((mins + 1))
  done
fi

# ── 3. Launch the remaining six ──────────────────────────────────────────────
for job in "${REST[@]}"; do
  launch_job "${job%%:*}" "${job##*:}"
done

# ── 4. Wait and summarize ────────────────────────────────────────────────────
say "all launches done — waiting for jobs to finish (days; safe to log out)..."
wait

done_n=0
for job in "${ALL_JOBS[@]}"; do
  if [ -f "$(folder_of "${job%%:*}" "${job##*:}")/global_stats.csv" ]; then
    done_n=$((done_n + 1))
  else
    say "[open  ] ${job%%:*} : ${job##*:} — not complete yet"
  fi
done
say "FINISHED: $done_n/7 target jobs have global_stats.csv."
[ "$done_n" -lt 7 ] && say "If jobs from an earlier invocation are still running, this is expected — rerun this script anytime to re-check or relaunch."
