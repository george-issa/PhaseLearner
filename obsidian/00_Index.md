# PhaseLearner — Obsidian Knowledge Base

Central index for the PhaseLearner research vault. Every important definition, decision, experiment, failure, proof obligation, source, checkpoint, stage/architecture revision, and next step is linked from here.

> Last sync: 2026-04-29 — pulled from `/Users/gissa/.claude/projects/-Users-gissa-Documents-SSH-PhaseLearner/memory/`, `RECOMMENDATIONS.txt`, `SIMULATION_UPDATES.txt`, and the current state of `ssh_ml_phase/`.
> Environment rule (applies everywhere in this vault): **always run `conda activate phaselearner` before any Python work.**

---

## Top-level documents

- [[01_Project_State]] — current snapshot: what's done, in-flight, blocked
- [[02_Definitions]] — glossary of physics, ML, and pipeline terms used everywhere else
- [[03_Source_Map]] — external references: papers, FSS values, dashboards, repos
- [[04_Claims_and_Proof_Obligations]] — every load-bearing claim and what would falsify it
- [[05_Code_Map]] — file-by-file tour of the package and ancillary scripts
- [[06_Experiments]] — runs and their numerical outcomes
- [[07_Failures_and_Negative_Results]] — what did NOT work and why it's recorded
- [[08_Decisions_and_Pivots]] — architectural / scientific decisions that closed off branches
- [[09_Authorship_and_Git_Policy]] — commit/push standards, no-Claude-attribution rule
- [[12_Long_Run_Justifications]] — when (and only when) a long run is allowed
- [[17_Memory_Health_and_Size]] — checklist for keeping this vault and the auto-memory tight

---

## Quick links by topic

### Definitions
- [[02_Definitions#α (alpha)]]
- [[02_Definitions#β (beta) and β_c]]
- [[02_Definitions#VBS / KVBS phase]]
- [[02_Definitions#Bond-bond V-shape artifact]]
- [[02_Definitions#Order parameters and observables]]
- [[02_Definitions#PCA susceptibility]]
- [[02_Definitions#UMAP sigmoid f_KVBS]]
- [[02_Definitions#Assaad / Greitemann–Iqbal CNN method]]
- [[02_Definitions#Autoencoder Var(z₀) susceptibility]]
- [[02_Definitions#Walker / bin / stride]]
- [[02_Definitions#Attractive Hubbard pilot terms]]

### Decisions & pivots
- [[08_Decisions_and_Pivots#Use Var(P₁) susceptibility, not KMeans, on autoencoder latents]]
- [[08_Decisions_and_Pivots#Drop BondBond0 single-correlator analysis]]
- [[08_Decisions_and_Pivots#UMAP on G(k) is current best β_c estimator]]
- [[08_Decisions_and_Pivots#Adopt per-α β_min floors instead of a global floor]]
- [[08_Decisions_and_Pivots#MPI walkers (4 × 250) instead of single 1000-bin chain]]
- [[08_Decisions_and_Pivots#Phonon extraction moved out of HMC inner loop]]
- [[08_Decisions_and_Pivots#Add radial-displacement observable]]
- [[08_Decisions_and_Pivots#Attractive-Hubbard pilot reads HDF5 directly, not stats CSV]]

### Experiments / checkpoints
- [[06_Experiments#Baseline PCA — BondBond012345 momentum]]
- [[06_Experiments#Baseline UMAP — BondBond012345 momentum]]
- [[06_Experiments#PCA / UMAP on G(k) (greens0_momentum)]]
- [[06_Experiments#bmin8 floor sweep]]
- [[06_Experiments#BondBond0 single correlator]]
- [[06_Experiments#electron_densities w1.00 (Holstein quick check)]]
- [[06_Experiments#Hubbard pilot smoke test (β=8)]]

### Failures & negative results
- [[07_Failures_and_Negative_Results#PCA Var(P₁) is monotone on G(k)]]
- [[07_Failures_and_Negative_Results#Autoencoder KMeans always lands at β≈12]]
- [[07_Failures_and_Negative_Results#BondBond0 single correlator collapses β_c]]
- [[07_Failures_and_Negative_Results#Old per-update phonon write produced 5.76M-row CSVs]]
- [[07_Failures_and_Negative_Results#Wrong Julia → numpy reshape (mode-major vs site-major)]]
- [[07_Failures_and_Negative_Results#Attractive-Hubbard stats CSV gives only mean ± std]]

### Proof obligations (open)
- [[04_Claims_and_Proof_Obligations#PO-1 Per-α β_min floors close UMAP G(k) gap]]
- [[04_Claims_and_Proof_Obligations#PO-2 Phonon radial displacement gives a clean monotone signal]]
- [[04_Claims_and_Proof_Obligations#PO-3 Assaad CNN converges with ≥1000 samples / β]]
- [[04_Claims_and_Proof_Obligations#PO-4 V-shape at β≈12 is observable artifact, not transition]]
- [[04_Claims_and_Proof_Obligations#PO-5 Hubbard pilot recovers β_c ≈ 8.5 with current pipeline]]

### Architecture revisions (stages)
- [[08_Decisions_and_Pivots#Stage A — single observable, single estimator]]
- [[08_Decisions_and_Pivots#Stage B — multi-observable, per-method estimators]]
- [[08_Decisions_and_Pivots#Stage C — per-α β_min, multi-walker DQMC, model registry]]
- [[08_Decisions_and_Pivots#Stage D (in flight) — second-model port to attractive Hubbard]]

### Sources
- [[03_Source_Map#FSS reference β_c (SSH oSSH, w=0.5)]]
- [[03_Source_Map#Attractive Hubbard T_c reference]]
- [[03_Source_Map#Bond Holstein paper (FSS β_c)]]
- [[03_Source_Map#SmoQyDQMC.jl]]
- [[03_Source_Map#Project files on cluster]]

### Next steps
- [[01_Project_State#Next steps — Priority A (no new DQMC)]]
- [[01_Project_State#Next steps — Priority B (1000-bin DQMC)]]
- [[01_Project_State#Next steps — Hubbard pilot]]

---

## Vault hygiene

- One concept per file; cross-link via wiki-links rather than duplicating prose.
- When a claim is invalidated, move it to [[07_Failures_and_Negative_Results]] with the date, do not silently delete.
- New experiments append rows to [[06_Experiments]]; new decisions append entries to [[08_Decisions_and_Pivots]].
- Re-run [[17_Memory_Health_and_Size]] checklist whenever this index passes ~150 lines or whenever a file feels stale.
