"""
Pipeline driver for ML phase detection — pure library, no CLI.

Top-level project scripts (scripts/run_all.py for computing,
scripts/run_plot.py for plot-only) build a config dict and call
:func:`run`. This module owns:

  1. Applying a model preset.
  2. Resolving OBSERVABLE → DATA_DIR + FILE_TEMPLATE for models with an
     OBSERVABLES registry ("ssh" and "attractive_hubbard").
  3. Building the run tag.
  4. Calling the requested method modules and the plotting routine.
"""
import os
from . import config as _cfg
from .data import make_tag


_VALID_METHODS = {"pca", "umap", "tsne", "assaad", "ae", "predict", "mean", "lbc"}


def run(*, model=None, observable=None, alphas=None,
        beta_min=None, beta_max=None, stride=None,
        file_template=None, data_dir=None, tag=None,
        methods=("pca", "umap"), plot=False, plot_only=False,
        alpha_sweep=False, betas=None, fss_alpha_c=None,
        latent_dim=2, n_seeds=3, umap_scale_kmeans=True,
        pca_cluster_method="raw_2d", pca_cluster_dims=None,
        n_target=None, transform=None, sid=None,
        perplexity=30, assaad_nskips=None):
    """
    Run the requested ML methods (and optionally plot) for the given config.

    Parameters
    ----------
    model : str or None
        Preset name from ``config.MODEL_CONFIGS``.
    observable : str or None
        Observable name from the active model's ``OBSERVABLES`` registry.
        When set, ``data_dir`` and ``file_template`` are auto-derived from
        the model's ``OBSERVABLE_DATA_PATTERN``.
    alphas : sequence of float, or None
        Alpha values to process. ``None`` uses the preset.
    beta_min, beta_max, stride : optional
        Beta-window and row-stride filters passed to the loader.
    file_template, data_dir : str, optional
        Explicit overrides; win over both ``model`` preset and ``observable``.
    tag : str, optional
        Output label. ``None`` → auto-generated from ``file_template`` +
        beta bounds + stride.
    methods : iterable of {"pca", "umap", "tsne", "assaad", "ae", "predict",
        "mean", "lbc"}
        Which method modules to run.
    plot : bool
        Generate PDFs after computing.
    plot_only : bool
        Skip computing; only re-render PDFs from existing ``.npz`` files.
    alpha_sweep : bool
        Switch into fixed-β / varying-α mode (PCA + UMAP only).
    betas, fss_alpha_c : alpha-sweep arguments.
    latent_dim, n_seeds : autoencoder arguments.
    umap_scale_kmeans : bool, default True
        Z-score the 2D UMAP embedding before KMeans.  See
        :func:`ssh_ml_phase.umap_method.run_umap_analysis`.
    pca_cluster_method : str, default "raw_2d"
        Post-PCA clustering recipe.  One of "raw_2d" (default), "spherical",
        "arcsinh", "log", "gmm".  See
        :func:`ssh_ml_phase.pca_method._cluster_pca_scores`.
    pca_cluster_dims : int or None, default None
        Number of leading PCs the recipe clusters on.  None → the recipe's
        natural default (1 for "arcsinh", 2 otherwise).
    perplexity : float, default 30
        t-SNE neighborhood size (typically 5–50).  Only used by ``"tsne"``.
    assaad_nskips : sequence of int, or None
        Guard-window sizes around each guessed β_c for the Assaad method.
        None → the method's default ``[4, 6, 8]``.  Only used by ``"assaad"``.
    n_target : float, optional
        Override the preset's ``N_TARGET`` (filling ⟨n⟩).  Rewrites the
        ``{n_target:.3f}`` placeholder in ``FILE_TEMPLATE`` and
        ``OBSERVABLE_DATA_PATTERN``.  Only meaningful for presets that
        define ``N_TARGET`` (currently ``attractive_hubbard``).
    sid : str, optional
        Output-folder ID.  When set, every output is nested under
        ``results/<sid>/`` (both ``npz/`` and ``figures/``) so different runs
        stay in separate, comparable folders.  ``None`` → default ``results/``.
        A plot-only run must use the same ``sid`` as the run that wrote the
        ``.npz`` files it reads.
    """
    bad = set(methods) - _VALID_METHODS
    if bad:
        raise ValueError(f"unknown method(s) {bad}; valid: {_VALID_METHODS}")

    # 1. apply model preset
    if model is not None:
        _cfg.apply_model(model, n_target=n_target)

    # 1b. sID — namespace every output under results/<sid>/ so different runs
    # (different changes) land in separate, comparable folders.  Set every call
    # so a None sid resets a previous run's nesting in the same process.
    _cfg.set_sid(sid)

    # data-loading transform (read lazily by data.load_beta_sweep); set every
    # call so it never leaks across runs in the same process.
    _cfg.TRANSFORM = transform

    # 2. observable → data_dir + file_template
    if observable is not None:
        valid = getattr(_cfg, "OBSERVABLES", None)
        pattern = getattr(_cfg, "OBSERVABLE_DATA_PATTERN", None)
        if valid is None or pattern is None:
            raise ValueError(
                f"model {model!r} has no OBSERVABLES registry; cannot resolve "
                f"OBSERVABLE={observable!r}."
            )
        if observable not in valid:
            raise ValueError(
                f"unknown OBSERVABLE {observable!r}. Valid for model "
                f"{model!r}: {valid}"
            )
        if data_dir is None:
            data_dir = os.path.join(
                _cfg.DATA_DIR,
                pattern["data_subdir"].format(obs=observable),
            )
        if file_template is None:
            file_template = pattern["file_template"].format(obs=observable)

    # 3. override DATA_DIR
    if data_dir is not None:
        _cfg.DATA_DIR = data_dir

    # 4. resolve alphas + file_template
    alphas        = list(alphas) if alphas else list(_cfg.ALPHAS)
    file_template = file_template or _cfg.FILE_TEMPLATE
    if alpha_sweep and file_template is None:
        file_template = getattr(_cfg, "ALPHA_SWEEP_FILE_TEMPLATE", None)
        if file_template is None:
            raise ValueError(
                "alpha-sweep needs FILE_TEMPLATE or ALPHA_SWEEP_FILE_TEMPLATE "
                "in the model preset"
            )

    # 5. tag + observable label
    tag = make_tag(beta_min, file_template, tag,
                   beta_max=beta_max, stride=stride)
    # keep transformed results in their own files (e.g. bond33_position_deltasusc)
    if transform:
        tag = f"{tag}_{transform}" if tag else transform
    obs_label = file_template.split("_L")[0] if file_template else ""
    obs_label = "_".join(s for s in obs_label.split("_") if "{" not in s)
    model_info = {
        "model":      model or "ssh",
        "observable": obs_label,
        "L":          _cfg.L,
        "beta_min":   beta_min,
        "beta_max":   beta_max,
        "tag":        tag,
        "transform":  transform,
    }

    # 6. header
    print("=" * 60)
    print(f"ML Phase Detection — model: {model or 'ssh'}  |  "
          f"mode: {'alpha-sweep' if alpha_sweep else 'beta-sweep'}")
    if sid:
        print(f"  sID           = {sid}  (outputs under {_cfg.RESULTS_DIR}/)")
    if tag:
        print(f"  tag           = {tag}  (results/npz/*_results_{tag}.npz)")
    print(f"  alphas        = {alphas}")
    print(f"  file_template = {file_template}")
    if data_dir is not None:
        print(f"  data_dir      = {data_dir}")
    if beta_min is not None: print(f"  beta_min      = {beta_min}")
    if beta_max is not None: print(f"  beta_max      = {beta_max}")
    if stride   is not None: print(f"  stride        = {stride}")
    print(f"  methods       = {list(methods)}"
          + ("  (plot-only — no compute)" if plot_only else ""))
    print("=" * 60)

    # 7. run methods
    if not plot_only:
        if alpha_sweep:
            if betas is None:
                raise ValueError("alpha-sweep mode requires `betas`")
            if "pca" in methods:
                print("\n>>> PCA alpha-sweep <<<")
                from .pca_method import run_all_alpha_sweep as pca_run
                pca_run(betas=betas, file_template=file_template,
                        fss_alpha_c=fss_alpha_c, tag=tag,
                        cluster_method=pca_cluster_method,
                        cluster_dims=pca_cluster_dims)
            if "umap" in methods:
                print("\n>>> UMAP alpha-sweep <<<")
                from .umap_method import run_all_alpha_sweep as umap_run
                umap_run(betas=betas, file_template=file_template,
                         fss_alpha_c=fss_alpha_c, tag=tag,
                         scale_kmeans=umap_scale_kmeans)
        else:
            if "pca" in methods:
                print("\n>>> PCA <<<")
                from .pca_method import run_all as pca_run
                pca_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max,
                        stride=stride, file_template=file_template, tag=tag,
                        cluster_method=pca_cluster_method,
                        cluster_dims=pca_cluster_dims)
            if "umap" in methods:
                print("\n>>> UMAP <<<")
                from .umap_method import run_all as umap_run
                umap_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max,
                         stride=stride, file_template=file_template, tag=tag,
                         scale_kmeans=umap_scale_kmeans)
            if "tsne" in methods:
                print("\n>>> t-SNE <<<")
                from .tsne_method import run_all as tsne_run
                tsne_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max,
                         stride=stride, file_template=file_template, tag=tag,
                         perplexity=perplexity, model=model)
            if "assaad" in methods:
                print("\n>>> Assaad (GI CNN) <<<")
                from .assaad_method import run_all as assaad_run
                assaad_run(alphas=alphas, tag=tag, n_seeds=n_seeds,
                           nskip_values=assaad_nskips,
                           beta_min=beta_min, beta_max=beta_max, stride=stride,
                           file_template=file_template)
            if "ae" in methods:
                print("\n>>> Autoencoder <<<")
                from .autoencoder import run_all as ae_run
                ae_run(alphas=alphas, latent_dim=latent_dim, n_seeds=n_seeds,
                       beta_min=beta_min, beta_max=beta_max, stride=stride,
                       file_template=file_template, tag=tag)
            if "predict" in methods:
                print("\n>>> Prediction-based (β regressor) <<<")
                from .prediction_method import run_all as predict_run
                predict_run(alphas=alphas, n_seeds=n_seeds,
                            beta_min=beta_min, beta_max=beta_max, stride=stride,
                            file_template=file_template, tag=tag)
            if "mean" in methods:
                print("\n>>> Mean-based (Arnold 2021) <<<")
                from .mean_method import run_all as mean_run
                mean_run(alphas=alphas,
                         beta_min=beta_min, beta_max=beta_max, stride=stride,
                         file_template=file_template, tag=tag)
            if "lbc" in methods:
                print("\n>>> Learning by Confusion (van Nieuwenburg 2017) <<<")
                from .lbc_method import run_all as lbc_run
                lbc_run(alphas=alphas, n_seeds=n_seeds,
                        beta_min=beta_min, beta_max=beta_max, stride=stride,
                        file_template=file_template, tag=tag)

    # 8. plot
    if plot or plot_only:
        print("\n>>> Plot <<<")
        from .plot import run_all_plots
        run_all_plots(
            tag           = tag,
            alphas        = alphas,
            fss_dict      = _cfg.FSS_DICT,
            fss_band_dict = getattr(_cfg, "FSS_BAND_DICT", None),
            model_info    = model_info,
            alpha_sweep   = alpha_sweep,
            betas         = betas,
            skip_umap     = "umap" not in methods,
        )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
