"""
Data loading utilities for SSH phase-detection ML methods.

Two loading modes are supported:

1. **Beta sweep** (default) — fixed alpha, varying beta.
   Each CSV row: [feature_0, ..., feature_{n-1}, beta]
   Used by: load_beta_sweep(alpha, ...)

2. **Alpha sweep** — fixed beta, varying alpha.
   Each CSV row: [feature_0, ..., feature_{n-1}, alpha]
   Used by: load_alpha_sweep(beta, ...)

The feature dimension is inferred automatically from the file, so the same
loader works for any observable (BondBond, GreenFunction, structure factor,
etc.) without code changes — just point FILE_TEMPLATE at the right files.
"""
import numpy as np
import os
from .config import (DATA_DIR, RESULTS_DIR,
                     FILE_TEMPLATE as DEFAULT_FILE_TEMPLATE,
                     BETA_MIN as DEFAULT_BETA_MIN,
                     BETA_MAX as DEFAULT_BETA_MAX,
                     BETA_MIN_PER_ALPHA, BETA_MAX_PER_ALPHA)


def load_beta_sweep(alpha, data_dir=None, file_template=None,
                    beta_min=None, beta_max=None, stride=None):
    """
    Load beta-sweep data for a given alpha value.

    Parameters
    ----------
    alpha : float
        Coupling constant (e.g. 0.240, 0.245, 0.250, 0.255).
    data_dir : str, optional
        Directory containing the CSV files. Defaults to DATA_DIR from config.
    file_template : str, optional
        Filename template with an ``{alpha:.4f}`` placeholder.
        Defaults to FILE_TEMPLATE from config.
    beta_min : float or None, optional
        Discard all samples with beta < beta_min.
        Priority: explicit arg > BETA_MIN_PER_ALPHA[alpha] > BETA_MIN global.
    beta_max : float or None, optional
        Discard all samples with beta > beta_max.
        Priority: explicit arg > BETA_MAX_PER_ALPHA[alpha] > BETA_MAX global.
    stride : int or None, optional
        Use every ``stride``-th row of the raw data (applied before beta
        filtering). stride=1 or None keeps all rows. Useful for subsampling
        dense DQMC datasets (e.g. stride=10 on 1000-bin data ≈ 100 effective
        samples per beta value).

    Returns
    -------
    X : ndarray, shape (N, n_features)
        Feature matrix. n_features is inferred from the file.
    betas : ndarray, shape (N,)
        Inverse temperature for each sample.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if file_template is None:
        file_template = DEFAULT_FILE_TEMPLATE

    # Per-alpha lookup with global fallback
    if beta_min is None:
        beta_min = BETA_MIN_PER_ALPHA.get(alpha, DEFAULT_BETA_MIN)
    if beta_max is None:
        beta_max = BETA_MAX_PER_ALPHA.get(alpha, DEFAULT_BETA_MAX)

    fname = os.path.join(data_dir, file_template.format(alpha=alpha))
    data = np.loadtxt(fname, delimiter=",")

    # Stride subsampling (applied to raw rows, before beta filtering)
    if stride is not None and stride > 1:
        data = data[::stride]

    X = data[:, :-1]       # all columns except last = features (any dimension)
    betas = data[:, -1]    # last column = beta

    if beta_min is not None:
        mask = betas >= beta_min
        X, betas = X[mask], betas[mask]
    if beta_max is not None:
        mask = betas <= beta_max
        X, betas = X[mask], betas[mask]

    return X, betas


def load_alpha_sweep(beta, data_dir=None, file_template=None,
                     alpha_min=None, alpha_max=None):
    """
    Load alpha-sweep data for a fixed beta value.

    Intended for DQMC datasets where beta is held fixed and alpha is swept,
    e.g. to locate the critical coupling alpha_c at a given temperature.

    Parameters
    ----------
    beta : float
        Fixed inverse temperature.
    data_dir : str, optional
        Directory containing the CSV files. Defaults to DATA_DIR from config.
    file_template : str
        Filename template with a ``{beta}`` placeholder, e.g.
        ``"BondBond_momentum_L12_w0.5_b{beta:.2f}.csv"``.
    alpha_min, alpha_max : float, optional
        Restrict the returned samples to alpha in [alpha_min, alpha_max].

    Returns
    -------
    X : ndarray, shape (N, n_features)
        Feature matrix. n_features is inferred from the file.
    alphas : ndarray, shape (N,)
        Coupling constant for each sample (last column of the CSV).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if file_template is None:
        raise ValueError(
            "file_template is required for alpha-sweep data. "
            "Example: 'BondBond012345_momentum_L12_w0.5_b{beta:.2f}.csv'"
        )

    fname = os.path.join(data_dir, file_template.format(beta=beta))
    data = np.loadtxt(fname, delimiter=",")
    X = data[:, :-1]
    alphas = data[:, -1]

    if alpha_min is not None:
        mask = alphas >= alpha_min
        X, alphas = X[mask], alphas[mask]
    if alpha_max is not None:
        mask = alphas <= alpha_max
        X, alphas = X[mask], alphas[mask]

    return X, alphas


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def make_tag(beta_min=None, file_template=None, tag=None,
             beta_max=None, stride=None):
    """
    Build a short string that uniquely identifies a run's data configuration.

    Priority:
      1. If ``tag`` is given explicitly, return it as-is.
      2. Otherwise auto-generate from the observable name (file_template),
         beta bounds, and stride.  Returns None for a fully default run.

    The observable name is extracted automatically from file_template — you
    do NOT need to pass ``--tag`` separately when ``--file-template`` is given.

    Examples
    --------
    >>> make_tag()             # "BondBond012345_momentum"  (from config default)
    >>> make_tag(beta_min=8.0) # "BondBond012345_momentum_bmin8"
    >>> make_tag(file_template="greens0_momentum_L12_a{alpha:.4f}.csv")
    ...                        # "greens0_momentum"
    >>> make_tag(file_template="greens0_momentum_L12_a{alpha:.4f}.csv",
    ...          beta_min=5, beta_max=20)  # "greens0_momentum_bmin5_bmax20"
    >>> make_tag(beta_min=8, stride=10)   # "BondBond012345_momentum_bmin8_stride10"
    >>> make_tag(tag="myrun")             # "myrun"
    """
    if tag is not None:
        return tag

    parts = []
    # Use the CLI-supplied template if given, otherwise fall back to the
    # config default so the observable name always appears in the tag.
    effective_template = file_template if file_template is not None else DEFAULT_FILE_TEMPLATE
    if effective_template is not None:
        # Extract observable label: everything up to the first "_L", then strip
        # any underscore-separated segments that contain a format placeholder
        # (e.g. "_a{alpha:.4f}") so Holstein tags stay readable.
        # e.g. "greens0_momentum_L12_..."                       → "greens0_momentum"
        #      "BondBond012345_momentum_L12_..."                → "BondBond012345_momentum"
        #      "electron_densities_w1.00_a{alpha:.4f}_L12-1.csv"→ "electron_densities_w1.00"
        base = effective_template.split("_L")[0]
        base = "_".join(seg for seg in base.split("_") if "{" not in seg)
        parts.append(base)
    if beta_min is not None:
        parts.append(f"bmin{beta_min:g}")
    if beta_max is not None:
        parts.append(f"bmax{beta_max:g}")
    if stride is not None and stride > 1:
        parts.append(f"stride{stride:g}")

    return "_".join(parts) if parts else None


def result_filename(method, tag=None):
    """
    Return the .npz filename for a given method and optional tag.

    Parameters
    ----------
    method : str
        One of ``"pca"``, ``"umap"``, ``"assaad"``, ``"autoencoder"``.
    tag : str or None
        Run tag produced by :func:`make_tag`.  None → no suffix.

    Examples
    --------
    >>> result_filename("pca")            # "pca_results.npz"
    >>> result_filename("pca", "bmin8.0") # "pca_results_bmin8.0.npz"
    """
    suffix = f"_{tag}" if tag else ""
    return f"{method}_results{suffix}.npz"
