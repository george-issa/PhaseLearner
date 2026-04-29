#!/usr/bin/env python3
"""
Run all ML methods for phase transition detection.

Usage:
    python3.11 -m ssh_ml_phase                                               # all methods, all alphas
    python3.11 -m ssh_ml_phase --model holstein --data-dir /path/to/data     # Holstein preset
    python3.11 -m ssh_ml_phase --model bond_holstein --data-dir /path/to/data
    python3.11 -m ssh_ml_phase --alphas 0.240                                # single alpha (fast)
    python3.11 -m ssh_ml_phase --beta-min 8.0                                # skip low-beta region
    python3.11 -m ssh_ml_phase --skip-umap                                   # skip UMAP (slow)
    python3.11 -m ssh_ml_phase --skip-assaad                                 # skip Assaad (slowest)
    python3.11 -m ssh_ml_phase --file-template "GF_L12_a{alpha:.4f}.csv"     # new observable
    python3.11 -m ssh_ml_phase --tag myrun                                   # explicit output label
    python3.11 -m ssh_ml_phase --plot                                        # run + generate PDFs
    python3.11 -m ssh_ml_phase --plot-only                                   # only generate PDFs (no run)
    python3.11 -m ssh_ml_phase --alpha-sweep --betas 0.35                    # alpha-sweep pipeline
    python3.11 -m ssh_ml_phase --alpha-sweep --betas 0.35 --plot             # alpha-sweep + PDFs
"""
import argparse


# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  —  edit here to run without using the command line
# ══════════════════════════════════════════════════════════════════════════════

MODEL        = "bond_holstein"  # preset: None | "ssh" | "holstein" | "bond_holstein"
ALPHAS       = [1.4142]         # None → use preset alphas; or e.g. [1.4142]
TAG          = None             # explicit output label (auto-generated if None)
BETA_MIN     = None             # discard samples with β < this value
BETA_MAX     = None             # discard samples with β > this value
STRIDE       = None             # use every N-th row (e.g. 10 for 1000-bin data)
FILE_TEMPLATE = None            # None → use preset template
DATA_DIR     = None             # None → use default data/ directory
SKIP_UMAP    = False
SKIP_ASSAAD  = True
SKIP_AE      = True
PLOT         = False            # generate PDFs after running
PLOT_ONLY    = False            # skip computation, generate PDFs from existing .npz
ALPHA_SWEEP  = False            # True → alpha-sweep pipeline
BETAS        = None             # fixed β value(s) for alpha-sweep, e.g. [0.35]
FSS_ALPHA_C  = None             # FSS α_c reference for alpha-sweep

# ══════════════════════════════════════════════════════════════════════════════


def main():
    # Import config early so apply_model() can be called before lazy method imports
    from . import config as _cfg

    parser = argparse.ArgumentParser(description="Run all ML phase detection methods")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Named model preset: 'ssh' (default), 'holstein', "
                             "'bond_holstein'. Sets alphas, file-template, and FSS "
                             "reference values. Explicit --alphas / --file-template "
                             "flags override the preset.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values to process (default: all from preset/config)")
    parser.add_argument("--beta-min", type=float, default=None,
                        help="Discard samples with beta < this value before analysis")
    parser.add_argument("--beta-max", type=float, default=None,
                        help="Discard samples with beta > this value before analysis")
    parser.add_argument("--stride", type=int, default=None,
                        help="Use every N-th row (e.g. --stride 10 on 1000-bin data "
                             "gives ~100 effective samples per beta)")
    parser.add_argument("--file-template", type=str, default=None,
                        help="Filename template with {alpha:.4f} placeholder "
                             "(overrides FILE_TEMPLATE in config/preset)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing input CSV files "
                             "(overrides DATA_DIR in config)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Label appended to every output .npz filename. "
                             "Auto-generated from observable + beta_min if omitted.")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--skip-assaad", action="store_true")
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of AE random seeds to average over")
    # ── Plotting flags ────────────────────────────────────────────────────────
    parser.add_argument("--plot", action="store_true",
                        help="Generate PDF figures after running all methods")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running methods; only generate PDFs from existing .npz files")
    # ── Alpha-sweep flags ─────────────────────────────────────────────────────
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Run alpha-sweep pipeline (fixed β, varying α) instead of beta-sweep")
    parser.add_argument("--betas", nargs="+", type=float, default=None,
                        help="Fixed β value(s) for alpha-sweep mode")
    parser.add_argument("--fss-alpha-c", type=float, default=None,
                        help="FSS α_c reference for alpha-sweep mode")
    args = parser.parse_args()

    # ── Merge: CLI args override the config block at the top of this file ────
    model       = args.model       or MODEL
    tag         = args.tag         or TAG
    beta_min    = args.beta_min    if args.beta_min    is not None else BETA_MIN
    beta_max    = args.beta_max    if args.beta_max    is not None else BETA_MAX
    stride      = args.stride      if args.stride      is not None else STRIDE
    alpha_sweep = args.alpha_sweep or ALPHA_SWEEP
    betas       = args.betas       or BETAS
    fss_alpha_c = args.fss_alpha_c if args.fss_alpha_c is not None else FSS_ALPHA_C
    skip_umap   = args.skip_umap   or SKIP_UMAP
    skip_assaad = args.skip_assaad or SKIP_ASSAAD
    skip_ae     = args.skip_ae     or SKIP_AE
    do_plot     = args.plot        or PLOT
    plot_only   = args.plot_only   or PLOT_ONLY
    data_dir    = args.data_dir    or DATA_DIR
    alphas_cli  = args.alphas      or ALPHAS   # may still be None → use preset
    ft_cli      = args.file_template or FILE_TEMPLATE

    # ── Apply model preset BEFORE lazy method imports ─────────────────────────
    if model is not None:
        _cfg.apply_model(model)

    # Override DATA_DIR at runtime if requested
    if data_dir is not None:
        _cfg.DATA_DIR = data_dir

    # Resolve alpha list and file template
    alphas        = alphas_cli or _cfg.ALPHAS
    file_template = ft_cli     or _cfg.FILE_TEMPLATE

    # For alpha-sweep, use ALPHA_SWEEP_FILE_TEMPLATE from preset if --file-template not given
    if alpha_sweep and args.file_template is None:
        asft = getattr(_cfg, "ALPHA_SWEEP_FILE_TEMPLATE", None)
        if asft:
            file_template = asft

    print("=" * 60)
    model_label = model if model else "ssh (default)"
    mode_label  = "alpha-sweep" if alpha_sweep else "beta-sweep"
    print(f"ML Phase Detection — model: {model_label}  |  mode: {mode_label}")
    # Resolve tag (auto-generate if not given explicitly)
    from .data import make_tag as _make_tag
    resolved_tag = _make_tag(beta_min, file_template, tag,
                             beta_max=beta_max, stride=stride)
    if resolved_tag:
        print(f"  tag = {resolved_tag}  → results will be saved as *_results_{resolved_tag}.npz")
    if model is not None:
        print(f"  alphas        = {alphas}")
        print(f"  file_template = {file_template}")
    if alpha_sweep and betas is not None:
        print(f"  betas (fixed) = {betas}")
    if beta_min is not None:
        print(f"  beta_min = {beta_min}  (low-beta samples excluded)")
    if beta_max is not None:
        print(f"  beta_max = {beta_max}  (high-beta samples excluded)")
    if stride is not None:
        print(f"  stride = {stride}  (using every {stride}-th row)")
    if data_dir is not None:
        print(f"  data_dir = {data_dir}")
    if plot_only:
        print("  plot-only: skipping computation, generating PDFs only")
    elif do_plot:
        print("  plot: will generate PDFs after computation")
    print("=" * 60)

    # ── Build model_info for plotting ─────────────────────────────────────────
    observable = file_template.split("_L")[0] if file_template else ""
    observable = "_".join(s for s in observable.split("_") if "{" not in s)
    model_info = {
        "model":      model or "ssh",
        "observable": observable,
        "L":          _cfg.L,
        "beta_min":   beta_min,
        "beta_max":   beta_max,
        "tag":        resolved_tag,
    }

    if not plot_only:
        if alpha_sweep:
            # ── Alpha-sweep: PCA + UMAP only (no Assaad/AE) ─────────────────
            if betas is None:
                print("ERROR: alpha-sweep mode requires BETAS to be set"); return

            print("\n>>> PCA alpha-sweep <<<")
            from .pca_method import run_all_alpha_sweep as pca_alpha_run
            pca_alpha_run(betas=betas, file_template=file_template,
                          fss_alpha_c=fss_alpha_c, tag=resolved_tag)

            if not skip_umap:
                print("\n>>> UMAP alpha-sweep <<<")
                from .umap_method import run_all_alpha_sweep as umap_alpha_run
                umap_alpha_run(betas=betas, file_template=file_template,
                               fss_alpha_c=fss_alpha_c, tag=resolved_tag)
            else:
                print("\n>>> UMAP skipped <<<")

        else:
            # ── Beta-sweep: all four methods ─────────────────────────────────

            # 1. PCA (fast)
            print("\n>>> PCA method <<<")
            from .pca_method import run_all as pca_run
            pca_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max, stride=stride,
                    file_template=file_template, tag=resolved_tag)

            # 2. UMAP (moderate)
            if not skip_umap:
                print("\n>>> UMAP method <<<")
                from .umap_method import run_all as umap_run
                umap_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max, stride=stride,
                         file_template=file_template, tag=resolved_tag)
            else:
                print("\n>>> UMAP skipped <<<")

            # 3. Assaad/GI (slow — trains many CNNs)
            if not skip_assaad:
                print("\n>>> Assaad (GI) method <<<")
                from .assaad_method import run_all as assaad_run
                assaad_run(alphas=alphas, tag=resolved_tag,
                           beta_min=beta_min, beta_max=beta_max, stride=stride,
                           file_template=file_template)
            else:
                print("\n>>> Assaad skipped <<<")

            # 4. Autoencoder (moderate)
            if not skip_ae:
                print("\n>>> Autoencoder method <<<")
                from .autoencoder import run_all as ae_run
                ae_run(alphas=alphas, latent_dim=args.latent_dim, n_seeds=args.n_seeds,
                       beta_min=beta_min, beta_max=beta_max, stride=stride,
                       file_template=file_template, tag=resolved_tag)
            else:
                print("\n>>> Autoencoder skipped <<<")

    # ── Plotting ──────────────────────────────────────────────────────────────
    if do_plot or plot_only:
        from .plot import run_all_plots
        run_all_plots(
            tag         = resolved_tag,
            alphas      = alphas,
            fss_dict    = _cfg.FSS_DICT,
            model_info  = model_info,
            alpha_sweep = alpha_sweep,
            betas       = betas,
            skip_umap   = skip_umap,
        )

    print("\n" + "=" * 60)
    print("All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
