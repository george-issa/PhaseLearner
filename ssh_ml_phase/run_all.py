#!/usr/bin/env python3
"""
Run all ML methods for phase transition detection.

Usage:
    python3.11 -m ssh_ml_phase                                              # all methods, all alphas
    python3.11 -m ssh_ml_phase --alphas 0.240                               # single alpha (fast)
    python3.11 -m ssh_ml_phase --beta-min 8.0                               # skip low-beta region
    python3.11 -m ssh_ml_phase --skip-umap                                  # skip UMAP (slow)
    python3.11 -m ssh_ml_phase --skip-assaad                                # skip Assaad (slowest)
    python3.11 -m ssh_ml_phase --file-template "GF_L12_a{alpha:.4f}.csv"    # new observable
    python3.11 -m ssh_ml_phase --tag myrun                                  # explicit output label
    python3.11 -m ssh_ml_phase --beta-min 8.0 --tag myrun                   # floor + explicit label
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run all SSH ML phase detection methods")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values to process (default: all)")
    parser.add_argument("--beta-min", type=float, default=None,
                        help="Discard samples with beta < this value before analysis")
    parser.add_argument("--beta-max", type=float, default=None,
                        help="Discard samples with beta > this value before analysis")
    parser.add_argument("--stride", type=int, default=None,
                        help="Use every N-th row (e.g. --stride 10 on 1000-bin data "
                             "gives ~100 effective samples per beta)")
    parser.add_argument("--file-template", type=str, default=None,
                        help="Filename template with {alpha:.4f} placeholder "
                             "(overrides FILE_TEMPLATE in config)")
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
    args = parser.parse_args()

    alphas = args.alphas
    beta_min = args.beta_min
    beta_max = args.beta_max
    stride = args.stride
    file_template = args.file_template
    tag = args.tag

    # Override DATA_DIR at runtime if requested
    if args.data_dir is not None:
        from . import config as _cfg
        _cfg.DATA_DIR = args.data_dir

    print("=" * 60)
    print("SSH ML Phase Detection — Running all methods")
    # Resolve tag (auto-generate if not given explicitly)
    from .data import make_tag as _make_tag
    resolved_tag = _make_tag(beta_min, file_template, tag,
                             beta_max=beta_max, stride=stride)
    if resolved_tag:
        print(f"  tag = {resolved_tag}  → results will be saved as *_results_{resolved_tag}.npz")
    if beta_min is not None:
        print(f"  beta_min = {beta_min}  (low-beta samples excluded)")
    if beta_max is not None:
        print(f"  beta_max = {beta_max}  (high-beta samples excluded)")
    if stride is not None:
        print(f"  stride = {stride}  (using every {stride}-th row)")
    if file_template is not None:
        print(f"  file_template = {file_template}")
    print("=" * 60)

    # 1. PCA (fast)
    print("\n>>> PCA method <<<")
    from .pca_method import run_all as pca_run
    pca_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max, stride=stride,
            file_template=file_template, tag=resolved_tag)

    # 2. UMAP (moderate)
    if not args.skip_umap:
        print("\n>>> UMAP method <<<")
        from .umap_method import run_all as umap_run
        umap_run(alphas=alphas, beta_min=beta_min, beta_max=beta_max, stride=stride,
                 file_template=file_template, tag=resolved_tag)
    else:
        print("\n>>> UMAP skipped <<<")

    # 3. Assaad/GI (slow — trains many CNNs)
    if not args.skip_assaad:
        print("\n>>> Assaad (GI) method <<<")
        from .assaad_method import run_all as assaad_run
        assaad_run(alphas=alphas, tag=resolved_tag,
                   beta_min=beta_min, beta_max=beta_max, stride=stride)
    else:
        print("\n>>> Assaad skipped <<<")

    # 4. Autoencoder (moderate)
    if not args.skip_ae:
        print("\n>>> Autoencoder method <<<")
        from .autoencoder import run_all as ae_run
        ae_run(alphas=alphas, latent_dim=args.latent_dim, n_seeds=args.n_seeds,
               beta_min=beta_min, beta_max=beta_max, stride=stride,
               file_template=file_template, tag=resolved_tag)
    else:
        print("\n>>> Autoencoder skipped <<<")

    print("\n" + "=" * 60)
    print("All done. Open SSH_ML_phase_detection.ipynb to visualize results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
