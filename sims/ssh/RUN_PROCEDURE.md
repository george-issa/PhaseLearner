# Running the 7 missing oSSH simulations — one command

Everything is baked into `run_missing_dell.sh` (data path, parameters, job
list, canary staging). Environment was set up 2026-06-10: `$HOME/oSSH_env`
(julia 1.8.5 + SmoQyDQMC 1.0.17), juliaup on PATH, MPICH_jll preference,
smoke test clean.

## 1. From the Mac — upload (once, after any script change)

```bash
cd ~/Documents/SSH/PhaseLearner
scp sims/ssh/run_missing_dell.sh sims/ssh/oSSH_array_runner.jl sims/ssh/RUN_PROCEDURE.md \
    gissa@physauth.physics.ucdavis.edu:/nfs/home/gissa/Documents/SSH/PhaseLearner/sims/ssh/
```

## 2. On the Dell — launch

```bash
cd ~/Documents/SSH/PhaseLearner/sims/ssh
nohup bash run_missing_dell.sh > driver.log 2>&1 &
```

That's it. The script then, on its own:

1. sanity-checks julia/env/data paths (aborts loudly if anything is off);
2. starts **β=14.00 as a canary** and waits for its first checkpoint
   (~25 min) before committing the machine to anything else;
3. launches the **other 6 jobs** (β=25.00 auto-resumes from its 74/100
   burn-in checkpoints — the driver log line says `[resume]`);
4. waits (~10 days) and prints a `FINISHED: n/7` summary.

## 3. Monitor (any time; logging out is fine)

```bash
tail driver.log                                       # driver narration
ls data/../data/ssh/sim_data/ossh_*-5/global_stats.csv 2>/dev/null | wc -l
#   ^ from repo root: 192 -> 199 as jobs finish
tail sims/ssh/oSSH_missing_*.output                   # per-job logs
ls -lt data/ssh/sim_data/ossh_*b23.00-5/checkpoint_*.jld2 | head -4
#   ^ checkpoint rotates every ~30-40 min while a job is alive
```

Timeline: β=14–16 jobs land around day 6–7; β=22.5/23/25 around day 9–10.

## 4. Crash / reboot / accidental kill

Rerun the exact same command (step 2). The script skips completed jobs,
detects still-running ones (won't double-launch), and resumes partial jobs
from their last checkpoint. To stop everything: `pkill -f oSSH_array_runner.jl`

## 5. When the summary says 7/7 (or the csv count hits 199)

```bash
cd ~/Documents/SSH/PhaseLearner/sims/ssh
env -u LD_LIBRARY_PATH julia +1.8.5 --project=$HOME/oSSH_env convert_jld2_csv_edited.jl
python3 ../../extract_phonons_edited.py --data-root /nfs/home/gissa/Documents/SSH/PhaseLearner/data/ssh/sim_data
```

Then sync the 7 new `ossh_*-5` folders down to the Mac (`data/ssh/sim_data/`)
and rerun the ML pipeline on the complete 199-point grid.
