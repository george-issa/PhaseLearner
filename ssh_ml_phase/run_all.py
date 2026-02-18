#!/usr/bin/env python3
"""
Run all ML methods for phase transition detection.

Usage:
    python3.11 -m ssh_ml_phase.run_all                     # all methods, all alphas
    python3.11 -m ssh_ml_phase.run_all --alphas 0.240      # single alpha (fast)
    python3.11 -m ssh_ml_phase.run_all --skip-umap         # skip UMAP (slow)
    python3.11 -m ssh_ml_phase.run_all --skip-assaad       # skip Assaad (slowest)
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run all SSH ML phase detection methods")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--skip-assaad", action="store_true")
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=2)
    args = parser.parse_args()

    alphas = args.alphas

    print("=" * 60)
    print("SSH ML Phase Detection — Running all methods")
    print("=" * 60)

    # 1. PCA (fast)
    print("\n>>> PCA method <<<")
    from .pca_method import run_all as pca_run
    pca_run(alphas=alphas)

    # 2. UMAP (moderate)
    if not args.skip_umap:
        print("\n>>> UMAP method <<<")
        from .umap_method import run_all as umap_run
        umap_run(alphas=alphas)
    else:
        print("\n>>> UMAP skipped <<<")

    # 3. Assaad/GI (slow — trains many CNNs)
    if not args.skip_assaad:
        print("\n>>> Assaad (GI) method <<<")
        from .assaad_method import run_all as assaad_run
        assaad_run(alphas=alphas)
    else:
        print("\n>>> Assaad skipped <<<")

    # 4. Autoencoder (moderate)
    if not args.skip_ae:
        print("\n>>> Autoencoder method <<<")
        from .autoencoder import run_all as ae_run
        ae_run(alphas=alphas, latent_dim=args.latent_dim)
    else:
        print("\n>>> Autoencoder skipped <<<")

    print("\n" + "=" * 60)
    print("All done. Open SSH_ML_phase_detection.ipynb to visualize results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
