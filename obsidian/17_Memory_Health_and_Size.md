# 17 — Memory Health and Size

Checklist for keeping this Obsidian vault and the Claude auto-memory at `/Users/gissa/.claude/projects/-Users-gissa-Documents-SSH-PhaseLearner/memory/` tight, fresh, and non-redundant. Linked from [[00_Index]].

## Vault inventory (current)

| File | Purpose | Last touched |
|------|---------|--------------|
| `00_Index.md` | central index, all wiki-links | 2026-04-29 |
| `01_Project_State.md` | snapshot, headline numbers, next steps | 2026-04-29 |
| `02_Definitions.md` | physics/ML/pipeline glossary | 2026-04-29 |
| `03_Source_Map.md` | external refs, FSS values, β grids | 2026-04-29 |
| `04_Claims_and_Proof_Obligations.md` | every load-bearing claim + how to falsify | 2026-04-29 |
| `05_Code_Map.md` | file-by-file tour | 2026-04-29 |
| `06_Experiments.md` | append-only run ledger | 2026-04-29 |
| `07_Failures_and_Negative_Results.md` | what didn't work + cause | 2026-04-29 |
| `08_Decisions_and_Pivots.md` | architectural / scientific decisions, stages | 2026-04-29 |
| `09_Authorship_and_Git_Policy.md` | no-AI-attribution, commit / push standards | 2026-04-29 |
| `12_Long_Run_Justifications.md` | short-runs-first rule | 2026-04-29 |
| `17_Memory_Health_and_Size.md` | this file | 2026-04-29 |

## Auto-memory inventory (mirror)

The Claude auto-memory at `/Users/gissa/.claude/projects/-Users-gissa-Documents-SSH-PhaseLearner/memory/` currently holds:

- `MEMORY.md` (index)
- `user_profile.md`
- `project_overview.md`
- `project_methods.md`
- `project_pending.md`
- `project_data_pipeline.md`
- `project_hubbard_pilot.md`
- `reference_tower_env.md`
- `feedback_command_execution.md`

Everything in those files is mirrored into this vault — **the auto-memory remains authoritative for cross-session recall**, while this vault is the human-readable knowledge base. If a fact diverges between the two, the most recently updated source wins; treat the other as stale until reconciled.

## Health rules

1. **One concept per file.** If a section grows into a self-contained essay, lift it into its own file and cross-link.
2. **`00_Index.md` is an index, not a memory.** One line per link. Keep it under ~150 lines so wiki-link autocompletion stays fast.
3. **Append, don't overwrite.**
   - New experiments append rows to [[06_Experiments]].
   - New decisions append to [[08_Decisions_and_Pivots]].
   - Failed approaches go to [[07_Failures_and_Negative_Results]] with a date and a one-line root cause; do not silently delete.
4. **Stale-claim flag.** Memories in the auto-memory carry a "this memory is N days old" reminder; treat any file:line citation older than ~14 days as something to verify against current code before quoting.
5. **No duplication of prose.** Wiki-link to definitions in [[02_Definitions]] rather than re-defining. If you find yourself defining `β_c` again, link instead.
6. **No raw data dumps.** Tables of numbers belong here only as a summary; the source is `results/*.npz` or the npz-loading notebook. Do not paste 1000-row CSV excerpts into the vault.

## Pruning checklist (run when the vault feels heavy)

- [ ] Any file > ~300 lines: split or move chunks to a child file.
- [ ] Any link in `00_Index.md` that no longer matches its destination: fix or remove.
- [ ] Any claim in [[04_Claims_and_Proof_Obligations]] that closed: move to its closed-section table.
- [ ] Any decision in [[08_Decisions_and_Pivots]] that was reversed: keep the original, add a follow-up entry recording the reversal — never delete the history.
- [ ] Any experiment in [[06_Experiments]] with no follow-through after 30 days: add a one-line note saying "abandoned" or "deferred" with the current date.
- [ ] `01_Project_State.md` headline numbers still match the latest `.npz` files in `results/`.

## Sync with auto-memory (when to do it)

Re-mirror this vault into the Claude auto-memory whenever:

- A claim in [[04_Claims_and_Proof_Obligations]] closes (move it).
- An experiment in [[06_Experiments]] flips a number in [[01_Project_State]]'s headline table.
- A decision in [[08_Decisions_and_Pivots]] is reversed.
- A new model preset is added to `MODEL_CONFIGS` (update [[02_Definitions]] *and* the auto-memory).

## Size budget

- Single file: ~300 lines maximum (split sooner rather than later).
- Index file `00_Index.md`: ~150 lines maximum.
- Whole vault: ~3000 lines maximum (we are well under this as of 2026-04-29).

## Out of scope for memory

These do **not** belong here:

- Code snippets that live in the package (`config.py`, `data.py`, etc.). Link to them in [[05_Code_Map]] instead.
- Git history (`git log` / `git blame` are authoritative).
- Raw `.npz` contents (load them in the notebook).
- Ephemeral todos for the current session — those live in the chat or in a TODO list, not in long-term memory.
