# 04 — Claims and Proof Obligations

Every load-bearing scientific claim about the pipeline, with the open proof obligation (PO) that would falsify or confirm it. New claims append to the table below; closed POs move their resolution date into the table and stay for the record.

## Open proof obligations

### PO-1 Per-α β_min floors close UMAP G(k) gap

**Claim:** UMAP's systematic 1–3 unit underestimate on G(k) is caused by low-β configurations contaminating the "ordered" KMeans cluster. Setting `BETA_MIN_PER_ALPHA = {0.240: 11, 0.245: 8, 0.250: 5, 0.255: 4}` will pull the UMAP β_c estimate within ≤ 1 unit of the FSS reference for at least 3 of 4 α values.

**How it would be falsified:** UMAP β_c with floors active is more than 1 unit below FSS for 2+ α values, OR the floor *over*-corrects (β_c > FSS) consistently — meaning the underestimate is not a low-β contamination effect but something else (e.g., embedding choice, cluster definition).

**How to test:** edit `ssh_ml_phase/config.py` `BETA_MIN_PER_ALPHA`, run

```bash
conda activate phaselearner
python -m ssh_ml_phase --skip-assaad --skip-ae \
  --file-template "greens0_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv" \
  --tag greens0_momentum_floored
```

Compare `umap_results_greens0_momentum_floored.npz` vs. `umap_results_greens0_momentum.npz`.

**Status:** open as of 2026-04-29.

---

### PO-2 Phonon radial displacement gives a clean monotone signal

**Claim:** The τ-averaged radial displacement `r[i] = mean_τ √(X[i,τ]² + Y[i,τ]²)` is a clean, domain-invariant order parameter. UMAP + ⟨|P₁|⟩-derivative on `radial_disp_*.csv` will give a tighter β_c estimate than G(k) (smaller σ across α) once 1000-bin DQMC data lands.

**How it would be falsified:** σ across α (0.240–0.255) of the radial-disp UMAP β_c estimate is *larger* than the corresponding G(k) UMAP estimate, or the bond-bond V-shape artifact reappears in radial-disp PCA.

**How to test:** when phonon CSVs land in `data/`,

```bash
conda activate phaselearner
python -m ssh_ml_phase.visualize \
  --file-template "radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
python -m ssh_ml_phase --skip-assaad --skip-ae \
  --file-template "radial_disp_L12_w0.5_m0.0_a{alpha:.4f}-1.csv"
```

**Status:** open; pending phonon DQMC runs.

---

### PO-3 Assaad CNN converges with ≥1000 samples / β

**Claim:** With 1000 DQMC samples per (α, β) (4 walkers × 250 bins), the Assaad / Greitemann–Iqbal CNN classifier sweep converges to a β_c estimate within ≤ 1 unit of the FSS reference for at least 3 of 4 α values.

**How it would be falsified:** test accuracy stays at chance (0.5 ± 0.05) across the candidate β grid for any α; OR the argmax-accuracy β candidate is > 1 unit off FSS for 2+ α.

**How to test:** once 1000-bin data lands,

```bash
conda activate phaselearner
python -m ssh_ml_phase.assaad_method   # remove --skip-assaad
```

**Status:** open; blocked on 1000-bin DQMC data.

---

### PO-4 V-shape at β≈12 is observable artifact, not transition

**Claim:** The bond-bond V-shape at β ≈ 12 for *all* α values is an observable artifact (likely tied to short-range bond correlations dominating before long-range order develops), not a real phase transition. Evidence: it appears at the same β regardless of α, while FSS β_c moves with α from 17.5 (α=0.240) down to 12.3 (α=0.255).

**How it would be falsified:** an independent observable (G(k), radial-disp) shows a feature at the *same* β ≈ 12 for at least one α where FSS β_c is not 12.

**How to test:** PCA / UMAP scans of G(k) and radial-disp; check whether any non-bond-bond observable shows structure at β=12 for, say, α=0.240 (where FSS β_c=17.5).

**Status:** open. So far G(k) does NOT show structure at β=12 — partially supports the claim.

---

### PO-5 Hubbard pilot recovers β_c ≈ 8.5 with current pipeline

**Claim:** The same `ssh_ml_phase` pipeline (PCA susceptibility + UMAP sigmoid) applied to the on-site s-wave pair susceptibility `P_s(q=0)` of the 2D attractive Hubbard model at U = −4, ⟨n⟩ = 0.875, L = 8 will recover a β_c estimate within ≤ 1 unit of the literature β_c ≈ 8.5.

**How it would be falsified:** PCA Var(P₁) and UMAP sigmoid β_c estimates differ from the Paiva et al. β_c ≈ 8.5 by > 1 unit, OR the s-wave pair signal is too noisy to give a clean susceptibility peak with 1000 samples / β.

**How to test:** smoke test already passed (P_s(q=0) ≈ 5.36 ± 0.37 at β=8 — ~10× SC enhancement vs. off-Γ values). Full β-sweep is queued; extractor `create_csv_pair_betasweep.py` is in place.

```bash
conda activate phaselearner
python -m ssh_ml_phase --model attractive_hubbard --skip-assaad --skip-ae
```

**Status:** open; pilot β-sweep submitted.

---

## Closed / superseded claims

(none yet — once a PO closes, copy its row here with the resolution date and a one-line outcome.)
