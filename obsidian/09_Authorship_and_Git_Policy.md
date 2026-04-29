# 09 — Authorship and Git Policy

The standing policy for commits, pushes, and authorship attribution on PhaseLearner. Linked from [[00_Index]].

## Authorship — hard rule

**Never add Claude (or any AI assistant) as a co-author or collaborator anywhere.** No `Co-Authored-By: Claude …`, no `Co-Authored-By: Anthropic …`, no comment trailers, no commit footers, no "Generated with …" tags, no PR-body attribution, no notebook cell metadata, no acknowledgements. The user is the sole author on every commit, push, PR, and figure caption produced from this repo.

If an automated workflow (a hook, a script, a CLI default) tries to add such a trailer, **strip it before pushing.**

## Commit / push messages

- **Human-readable**: a person reading `git log --oneline` should know what changed in plain English.
- **Intuitive**: lead with the *what* in the imperative. "Add per-α β_min floors", not "feat(config): introduce parametric beta-floor lookup".
- **Brief**: subject ≤ 70 characters; body only when the subject can't stand alone. No multi-paragraph bodies for one-line changes.
- **No fluff**: no ticket prefixes we don't actually use, no emoji, no auto-generated boilerplate.
- **Examples that fit the style:**
  - `Add per-α β_min floors to config.py`
  - `Fix wrong reshape in phonon extractor (mode-major)`
  - `Hubbard pilot: HDF5 extractor + smoke test`
  - `Drop BondBond0 single-correlator runs from default sweep`

## Push cadence

- **Push cleanly and often.** Small, self-contained commits; push after each one once it builds and the basic sanity check passes.
- **Don't bundle.** Pipeline change + new observable + plot tweak = three commits, not one.
- **Don't sit on local work.** A day's worth of edits should be on the remote by end-of-day unless the branch is intentionally a staging area.
- **Never force-push to `main`.**

## Coding standards

- **Clean repo.** No `%OUTDIR%`-style leftover sentinel directories, no unused branches in conditionals, no commented-out blocks left to rot.
- **Intuitive code.** Variables and functions named after what they *mean* (`mean_P1_deriv_bc`, `radial_disp`), not what they *do mechanically* (`func_calc1`).
- **Self-consistent.** Same convention everywhere: `{alpha:.4f}` placeholder, auto-tag from file template, model registry rather than per-model branches.
- **Clever code and insights.** Reach for the right primitive (UMAP for monotone observables, Var(P₁) for non-monotone). When a clever observation lands (e.g. radial-displacement collapses domain redundancy), record it in [[08_Decisions_and_Pivots]] so it isn't relitigated.
- **No dead weight.** Delete superseded files instead of keeping zombie variants. If a `_legacy` artifact must stick around, mark its successor in a one-line comment.
- **No AI attribution in code or comments.** This is the same rule as the authorship rule: code comments must read as the user's own.

## What goes in `.gitignore`

- `data/` raw CSVs over a few hundred MB (we already have 1000-row × 577-col phonon files coming).
- `results/*.npz` outputs are tracked by default but rotate when they balloon.
- Local environment artefacts: `Manifest.toml` and `LocalPreferences.toml` are repo-tracked because they pin the Julia env; per-machine venv directories under `~/Hubbard_2d/<venv-dir>/` are NOT.
- `__pycache__/`, `.ipynb_checkpoints/`, `.venv/` — always ignored.

## Pre-push checklist

1. `git status` — no stray untracked artifacts.
2. Subject line in the imperative, ≤ 70 chars.
3. No `Co-Authored-By:` trailer of any kind.
4. If it's a pipeline change, a quick smoke run with `--alphas 0.240 --skip-assaad --skip-ae` finishes without error.
5. Push.

## When in doubt

Default to fewer co-authors (i.e. only you), more commits, shorter messages.
