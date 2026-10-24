#!/usr/bin/env python3
"""
PhaseLearner — single entry point.

    conda activate phaselearner
    python scripts/run_all.py

Three sections to edit:

    1 · DATA    — MODEL + its PRESETS block (observable, α range, β window, …).
                  Shared by every method, so they're all compared on the same
                  data, plus the output switches (PLOT, TAG, SID, TRANSFORM).
    2 · METHODS — one independent block per method.  Flip its `run` on/off and
                  set that method's own knobs.  Each method is controlled alone.
    3 · RUN     — nothing to edit.

Outputs:
    results/npz/<method>_results_<tag>.npz
    results/figures/<model>/<method>/<tag>/      (only when PLOT = True)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssh_ml_phase.pipeline import run
from ssh_ml_phase.config import list_observables


# ════════════════════════════════════════════════════════════════════════════
#  1 · DATA  (shared by every method)
# ════════════════════════════════════════════════════════════════════════════
# ssh | attractive_hubbard | holstein | bond_holstein
MODEL = "ssh"

PLOT  = True      # render PDFs after computing
TAG   = None      # output label; None → auto-named from the observable
SID   = 1         # nest outputs under results/<SID>/{npz,figures}; None → results/

# optional data transform for position/momentum correlators (None keeps raw data):
#   "noorigin"  drop the origin / self-term column (r=0, or Γ)
#   "deltasusc" no-origin, per-sample median-subtracted, ÷ max|dev| (symmetric
#               [-1,+1]) → ~0 in the disordered phase, period-3 checkerboard
#               rising toward +1 through the transition (real-space / position).
#               Best on integrated (χ) position correlators: highest SNR.
TRANSFORM = None

#   `observables` accepts either an explicit list of names, or a keyword
#   (ssh / attractive_hubbard only): "all" / "equal_time" / "integrated".
PRESETS = {

    # ── optical SSH — honeycomb lattice, KVBS transition ────────────────────
    #   observable options (list as many as you want in `observables`):
    #     BondBond012345_momentum                     KVBS order parameter
    #     bond{33,44,55,34,35,45}_{position,momentum} individual bond-bond pairs
    #     greens{11,22,12}_{position,momentum}        Green's-function pairs
    #     integrated_<any of the above>               β-integrated susceptibility
    "ssh": dict(
        observables = [
            # "integrated_bond33_position",
            "integrated_bond33_momentum",
            # "integrated_bond44_position", "integrated_bond55_position",
            # "integrated_bond34_position", "integrated_bond35_position",
            # "integrated_bond45_position",
        ],                 # ← or a keyword: "all" / "equal_time" / "integrated"
        alphas   = [0.240],   # None → 0.240, 0.245, 0.250, 0.255
        beta_min = None,   # drop samples with β below this
        beta_max = None,   # drop samples with β above this
        stride   = None,   # use every Nth row (e.g. 20 to speed up t-SNE)
    ),

    # ── attractive Hubbard — square lattice, s-wave SC transition ───────────
    #   observable options (list as many as you want in `observables`):
    #     {density,pair}_{position,momentum}
    #     integrated_<any of the above>               β-integrated susceptibility
    "attractive_hubbard": dict(
        observables = [
            "integrated_pair_momentum",
            # "pair_momentum", "integrated_density_momentum",
        ],                 # ← or replace the list with "all" / "equal_time" / "integrated"
        n_target = 0.875,  # filling ⟨n⟩   (available: 0.500, 0.875)
        alphas   = None,   # None → 4.0    (for this model α holds |U|)
        beta_min = None,
        beta_max = None,
        stride   = None,
    ),

    # ── Holstein / bond-Holstein — no observable registry ───────────────────
    #   These models have no per-observable CSVs, so use [None] and the
    #   model's built-in FILE_TEMPLATE is used.
    "holstein": dict(
        observables = [None],
        alphas   = None,   # None → 0.6325, 0.7071, 0.8165, 1.0, 1.4142
        beta_min = None,
        beta_max = None,
        stride   = None,
    ),
    "bond_holstein": dict(
        observables = [None],
        alphas   = None,   # None → 0.6325, 0.7071, 0.8165, 1.0, 1.4142
        beta_min = None,
        beta_max = None,
        stride   = None,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
#  2 · METHODS  —  one independent block each: flip `run`, set its own knobs
# ════════════════════════════════════════════════════════════════════════════

# ── PCA ─────────────────────────────────────────────────────────────────────
PCA = dict(
    run            = False,
    cluster_method = "raw_2d",   # raw_2d | spherical | arcsinh | log | gmm
    cluster_dims   = 2,          # leading PCs the recipe clusters on (None → default)
)

# ── UMAP ────────────────────────────────────────────────────────────────────
UMAP = dict(
    run          = False,
    scale_kmeans = True,         # z-score the 2D embedding before KMeans
)

# ── t-SNE ───────────────────────────────────────────────────────────────────
TSNE = dict(
    run        = False,
    perplexity = 30.0,           # neighborhood size (typ. 5–50; larger → broader structure)
)

# ── Prediction (β regressor) ────────────────────────────────────────────────
PREDICT = dict(
    run     = False,
    n_seeds = 1,                 # random-init seeds averaged (1 → no error bars)
)

# ── Mean (Arnold 2021) ──────────────────────────────────────────────────────
MEAN = dict(
    run = False,                 # no method-specific knobs
)

# ── Learning by Confusion (van Nieuwenburg 2017) ────────────────────────────
LBC = dict(
    run     = False,
    n_seeds = 1,                 # random-init seeds averaged (1 → no error bars)
)

# ── Assaad / Greitemann–Iqbal ───────────────────────────────────────────────
ASSAAD = dict(
    run     = True,
    n_seeds = 1,                 # random-init seeds averaged (1 → no error bars)
    nskips  = [8],         # guard-window sizes around each guessed β_c
)

# ── Autoencoder ─────────────────────────────────────────────────────────────
AE = dict(
    run        = False,
    latent_dim = 2,
    n_seeds    = 1,
)


# ════════════════════════════════════════════════════════════════════════════
#  3 · RUN  (nothing to edit below)
# ════════════════════════════════════════════════════════════════════════════
# Ordered (name, config block); order only affects print/plot order.
_METHOD_BLOCKS = [
    ("pca", PCA), ("umap", UMAP), ("tsne", TSNE), ("predict", PREDICT),
    ("mean", MEAN), ("lbc", LBC), ("assaad", ASSAAD), ("ae", AE),
]


def _method_knobs(name, c):
    """Translate a method's config block into pipeline.run keyword arguments."""
    if name == "pca":
        return dict(pca_cluster_method=c["cluster_method"],
                    pca_cluster_dims=c["cluster_dims"])
    if name == "umap":
        return dict(umap_scale_kmeans=c["scale_kmeans"])
    if name == "tsne":
        return dict(perplexity=c["perplexity"])
    if name == "ae":
        return dict(latent_dim=c["latent_dim"], n_seeds=c["n_seeds"])
    if name == "assaad":
        return dict(n_seeds=c["n_seeds"], assaad_nskips=c["nskips"])
    if name in ("predict", "lbc"):
        return dict(n_seeds=c["n_seeds"])
    return {}   # mean — no method-specific knobs


enabled      = [(name, c) for name, c in _METHOD_BLOCKS if c.get("run")]
method_names = [name for name, _ in enabled]

cfg         = dict(PRESETS[MODEL])
observables = cfg.pop("observables") or [None]

# A keyword ("all" / "equal_time" / "integrated") expands to the matching names
# from the model's registry; an explicit list is used as given.
if isinstance(observables, str):
    observables = list_observables(MODEL, observables)

if not enabled:
    print("No methods enabled — set run=True in at least one method block (section 2).")

for observable in observables:
    if observable:
        print(f"\n{'#' * 70}\n#  observable: {observable}\n{'#' * 70}")
    # Each observable's results are auto-named after it, so one fixed TAG would
    # overwrite the others — only honour TAG when there's a single observable.
    tag = TAG if len(observables) == 1 else None

    # Compute each enabled method with its own knobs.  Plot once at the end so
    # methods aren't re-plotted as their .npz files accumulate.
    for name, c in enabled:
        run(model=MODEL, observable=observable, methods=[name], plot=False,
            tag=tag, transform=TRANSFORM, sid=SID,
            **_method_knobs(name, c), **cfg)

    if PLOT and enabled:
        run(model=MODEL, observable=observable, methods=method_names,
            plot_only=True, tag=tag, transform=TRANSFORM, sid=SID, **cfg)
