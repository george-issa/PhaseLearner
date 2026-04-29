# 01 — Project State

> Snapshot date: 2026-04-29

## One-line status

PhaseLearner detects the VBS β_c on the optical-SSH honeycomb lattice via four ML methods on DQMC data; UMAP on G(k) is the current best estimator (~1–3 units low without per-α β_min floors). A second-model port to the 2D attractive Hubbard SC transition is in pilot.

## Goal

Detect the critical inverse temperature β_c of the VBS (Valence Bond Solid) phase transition in the optical SSH model on an `L = 12` honeycomb lattice using unsupervised / semi-supervised ML on DQMC data — see [[02_Definitions#VBS / KVBS phase]] and [[03_Source_Map#FSS reference β_c (SSH oSSH, w=0.5)]].

## Fixed physical parameters

- Phonon frequency `ω = 0.5`, mass term `m = 0.0`, lattice `L = 12` (honeycomb).
- α (e-ph coupling) sweep: 0.240, 0.245, 0.250, 0.255 (core); 0.235, 0.2415, 0.2435 planned.
- β grid: depends on α — see `SIMULATION_UPDATES.txt` §2 (and the per-α β grid in [[03_Source_Map#Project files on cluster]]).

## Current numerical results — β_c estimates vs. FSS

Source: `RECOMMENDATIONS.txt` table; methods documented in [[05_Code_Map#ssh_ml_phase package]].

| Method                                   | α=0.240 | α=0.245 | α=0.250 | α=0.255 | Verdict |
|------------------------------------------|---------|---------|---------|---------|---------|
| **FSS reference**                        | 17.5    | 14.3    | 12.5    | 12.3    | ground truth |
| PCA  BondBond012345 (default)            | 17.0    | 15.5    | 14.0    | 19.0    | best PCA, fails at α=0.255 |
| PCA  BondBond012345 bmin8                | 17.0    | 15.5    | 14.0    | 19.0    | floor-insensitive at b=8 |
| PCA  BondBond0                           | 7.0     | 10.0    | 8.5     | 9.5     | abandoned ([[08_Decisions_and_Pivots#Drop BondBond0 single-correlator analysis]]) |
| PCA  greens0_momentum (Var P₁)           | 23.0    | 22.0    | 22.0    | 23.0    | broken — Var(P₁) monotone |
| PCA  greens0 mean P₁ derivative          | 21.8    | 19.2    | 15.2    | 18.8    | needs per-α β_min |
| UMAP BondBond012345 (default)            | 12.5    | 13.6    | 12.6    | 13.3    | dragged to artifact β≈12 |
| UMAP BondBond012345 bmin8                | 15.7    | 15.2    | 14.3    | 15.5    | clear improvement |
| **UMAP greens0_momentum**                | **14.7**| **13.4**| **11.7**| **10.9**| **best overall** |

Best single estimator right now: **UMAP on G(k)** with per-α β_min floors planned ([[04_Claims_and_Proof_Obligations#PO-1 Per-α β_min floors close UMAP G(k) gap]]).

## In-flight work

- 277 SLURM jobs scaling every (α, β) to 1000 samples (4 walkers × 250 bins) — see [[06_Experiments]] section "1000-bin DQMC sweep".
- Attractive-Hubbard SC pilot — smoke test passed (P_s(0) ≈ 5.36 ± 0.37 at β=8, U=−4, ⟨n⟩=0.875, L=8); β-sweep is queued. Details: [[02_Definitions#Attractive Hubbard pilot terms]] and [[03_Source_Map#Attractive Hubbard T_c reference]].

## Blockers

- Cluster wall-clock time. 277 jobs × wall-clock — see [[12_Long_Run_Justifications]] for the rule (short runs first, long runs only with explicit justification).
- α=0.240 β grid currently tops at β=23 although β_c=17.5; need to extend to β≈30 for a deeply ordered tail to anchor UMAP clusters.

## Next steps — Priority A (no new DQMC)

1. Edit [[05_Code_Map#config.py]] `BETA_MIN_PER_ALPHA`:
   - G(k):     `{0.240: 11, 0.245: 8, 0.250: 5, 0.255: 4}`
   - BondBond: `{0.240: 10, 0.245: 8, 0.250: 5, 0.255: 5}`
   Then re-run PCA + UMAP on each observable. Expected: UMAP G(k) closes most of the underestimate gap.
2. Run UMAP on `greens0_position` (real-space G(r)) — should be a 5-minute experiment ([[12_Long_Run_Justifications#Default — short runs]]).
3. Compare BondBond012345 with vs. without floors using `--tag` to avoid clobbering result `.npz`.

## Next steps — Priority B (1000-bin DQMC)

1. Extend α=0.240 β grid to β≈28–30.
2. Backfill all (α, β) to 1000 bins/point via 4 walkers × 250 bins (already coded; jobs queued).
3. Test Assaad CNN once ≥1000 samples/β are available — closes [[04_Claims_and_Proof_Obligations#PO-3 Assaad CNN converges with ≥1000 samples / β]].
4. Add α=0.2415, 0.2435 (interpolated FSS), and α=0.235 (deep-disordered anchor).
5. Phonon observables: tau=0 field (576 features) and radial-disp (288 features). Expected: cleanest VBS signal — closes [[04_Claims_and_Proof_Obligations#PO-2 Phonon radial displacement gives a clean monotone signal]].

## Next steps — Hubbard pilot

- Run β-sweep extractor (`create_csv_pair_betasweep.py`) once all `b{β}` folders complete.
- Run pipeline: `python -m ssh_ml_phase --model attractive_hubbard --skip-assaad --skip-ae` (after `conda activate phaselearner`).
- Compare PCA susceptibility peak and UMAP sigmoid β_c to reference T_c ≈ 0.118 t (β_c ≈ 8.5).
- Then extend to L=10, 12 for finite-size scaling on s-wave pair susceptibility.
