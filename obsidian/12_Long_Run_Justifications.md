# 12 — Long-Run Justifications

The standing rule on when a long run (DQMC sweep, 1000-bin extension, multi-α scan) is allowed and when it is not. Linked from [[00_Index]].

## Default — short runs

**Always prioritize short runs for quick tests and experiments.** A short run is anything that:

- finishes in under ~10 minutes wall-clock for ML, or under ~1 hour for DQMC,
- uses **one** α value, or a small subset (e.g. just α=0.240),
- uses a coarse β grid or a single β,
- uses `--stride 10` or higher to subsample 1000-bin data,
- uses fewer SLURM jobs (e.g. one alpha's β grid, not all 277 jobs).

Short runs are the default. **Pose every new test as "what's the smallest run that would tell me yes/no on this question?" before scaling up.**

Examples of correct first runs:

```bash
conda activate phaselearner

# Single alpha, all methods skipped except UMAP — minutes.
python -m ssh_ml_phase --alphas 0.240 --skip-assaad --skip-ae --skip-pca \
  --file-template "greens0_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"

# 1/10 sample density on 1000-bin data to confirm the pipeline still converges.
python -m ssh_ml_phase --stride 10 --skip-assaad

# Visualize a new observable BEFORE running ML (saves a wasted sweep).
python -m ssh_ml_phase.visualize --alphas 0.240 \
  --file-template "radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"

# Hubbard pilot smoke run, single β.
python -m ssh_ml_phase --model attractive_hubbard --alphas 4.0 \
  --beta-min 8 --beta-max 8 --skip-assaad --skip-ae
```

## When a long run is allowed

A long run (full α grid × β grid; 1000-bin DQMC; AE training across all α; CNN sweep) is allowed only when **all** of these hold:

1. **A short run already ran and the result is consistent.** No surprises that would change the design.
2. **There is an open proof obligation in [[04_Claims_and_Proof_Obligations]] that the long run actually closes.** "We should re-run everything" is not a justification.
3. **The expected result is written down before launching** (so we can tell ex-post whether the run succeeded or surprised us).
4. **The cluster wall-clock budget can absorb the run without blocking other work.**

Mark a long run as such with `--tag long` (or a more specific tag) so its result file is unambiguous.

## When a long run is NOT allowed

- "Let's just re-run with a different floor." → Run **one α** first, compare, then decide.
- "Maybe more samples will fix it." → Use `--stride` on existing 1000-bin data first.
- "Let me kick off all observables in parallel." → Run **the most promising** observable first; queue the rest only after the result is informative.
- "Let me ramp from L=8 → L=10 → L=12 on the Hubbard pilot now." → Land L=8 first; FSS comes after the L=8 pipeline result matches the literature reference.
- Re-running because a small flag changed and you want a clean overwrite. → Use `--tag` to disambiguate, keep both, decide later.

## Cluster etiquette

- The 277-job sweep counts as a long run; submit only after Priority A (per-α floors, `greens0_position` UMAP) has confirmed the pipeline picks up the expected effects.
- If a long run is in flight, don't queue another long run on top of it for the same observable — let the first finish, look at the result, then plan the next.
- Always tag the result file (`--tag 1000bin`, `--tag floored`, etc.) so downstream notebook plots cite the exact provenance.

## Standing escape clause

If a long run is genuinely the only path forward (e.g. Assaad CNN cannot be tested with stride alone), say so explicitly in the commit message and in [[06_Experiments]] when logging the run. Long runs are not forbidden — just not the default.
