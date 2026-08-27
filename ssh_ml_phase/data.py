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
from . import config as _cfg

# Module attributes are looked up lazily through `_cfg` so that runtime mutations
# (apply_model() and CLI overrides like --data-dir / --file-template) take effect
# in callers that imported this module before the override happened.


def apply_transform(X, kind):
    """Origin-excluded variants of a correlator feature matrix.

    The origin is the self-term (r=0 in position space, Γ in momentum space):
    the largest-mean column, which carries the disconnected background and
    swamps the physics.  Both variants drop it.

    kind
        "noorigin"  — drop the origin column only.
        "deltasusc" — drop the origin, then subtract each sample's spatial
                      median (the paper's median-background convention, which
                      lands on the suppressed 2/3 plateau) and divide by the
                      global maximum absolute deviation ``max|dev|`` over all
                      samples.  This is the paper's normalisation: the map is
                      symmetric in [-1, +1], ≈0 in the disordered phase, with
                      the period-3 cells rising toward +1 through the transition.
    """
    origin = int(np.argmax(X.mean(axis=0)))
    X = np.delete(X, origin, axis=1)
    if kind == "noorigin":
        return X
    if kind == "deltasusc":
        delta = X - np.median(X, axis=1, keepdims=True)
        return delta / np.abs(delta).max()
    raise ValueError(
        f"unknown transform {kind!r}; use 'noorigin' or 'deltasusc'."
    )


def lattice_snapshots(X, betas, n=3):
    """Provenance maps of the exact features fed to an ML method.

    Returns ``(snap_betas, grids)`` where ``grids`` has shape ``(m, L, L)`` —
    the per-beta mean of the input features at up to ``n`` evenly spaced betas
    (coldest, middle, hottest), reshaped onto the L×L cell grid (indexed
    ``[r2, r1]``).  Origin-excluded inputs (``L*L - 1`` features) get the origin
    cell reinserted as NaN so the map stays square.  Returns ``(None, None)``
    when the feature count is neither ``L*L`` nor ``L*L - 1`` (e.g. phonon /
    radial inputs are not single lattices).
    """
    L  = _cfg.L
    nf = X.shape[1]
    if nf == L * L:
        to_grid = lambda v: v.reshape(L, L)
    elif nf == L * L - 1:                      # origin (index 0) was excluded
        def to_grid(v):
            full = np.full(L * L, np.nan)
            full[1:] = v
            return full.reshape(L, L)
    else:
        return None, None
    ub   = np.unique(betas)
    pick = ub[np.unique(np.linspace(0, len(ub) - 1, min(n, len(ub))).round().astype(int))]
    grids = np.array([to_grid(X[betas == b].mean(axis=0)) for b in pick])
    return pick, grids


def store_snapshots(save_dict, key, result):
    """Copy a result dict's input-lattice snapshots into a method's save dict."""
    if result.get("inlat_grids") is not None:
        save_dict[f"{key}_inlat_betas"] = result["inlat_betas"]
        save_dict[f"{key}_inlat_grids"] = result["inlat_grids"]


def load_beta_sweep(alpha, data_dir=None, file_template=None,
                    beta_min=None, beta_max=None, stride=None, transform=None):
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
    transform : str or None, optional
        Apply an origin-excluded variant to the features (see
        :func:`apply_transform`): ``"noorigin"`` or ``"deltasusc"``.
        Defaults to ``config.TRANSFORM``.  The global-max normalisation in
        ``"deltasusc"`` is taken over all rows of this alpha file.

    Returns
    -------
    X : ndarray, shape (N, n_features)
        Feature matrix. n_features is inferred from the file.
    betas : ndarray, shape (N,)
        Inverse temperature for each sample.
    """
    if data_dir is None:
        data_dir = _cfg.DATA_DIR
    if file_template is None:
        file_template = _cfg.FILE_TEMPLATE
    if transform is None:
        transform = getattr(_cfg, "TRANSFORM", None)

    # Per-alpha lookup with global fallback
    if beta_min is None:
        beta_min = _cfg.BETA_MIN_PER_ALPHA.get(alpha, _cfg.BETA_MIN)
    if beta_max is None:
        beta_max = _cfg.BETA_MAX_PER_ALPHA.get(alpha, _cfg.BETA_MAX)

    fname = os.path.join(data_dir, file_template.format(alpha=alpha))
    data = np.loadtxt(fname, delimiter=",")

    # Stride subsampling (applied to raw rows, before beta filtering)
    if stride is not None and stride > 1:
        data = data[::stride]

    X = data[:, :-1]       # all columns except last = features (any dimension)
    betas = data[:, -1]    # last column = beta

    if transform:
        X = apply_transform(X, transform)

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
        data_dir = _cfg.DATA_DIR
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
    os.makedirs(_cfg.RESULTS_DIR, exist_ok=True)
    return _cfg.RESULTS_DIR


def ensure_npz_dir():
    """Create the npz subdir of the results directory if it doesn't exist."""
    os.makedirs(_cfg.NPZ_DIR, exist_ok=True)
    return _cfg.NPZ_DIR


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
    effective_template = file_template if file_template is not None else _cfg.FILE_TEMPLATE
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
        # Surface ⟨n⟩ from the template (Hubbard naming: "..._L8_n0.500_U...").
        # Without this, n=0.5 and n=0.875 runs would share auto-tags and
        # overwrite each other's npz files.
        import re
        m = re.search(r"_n(\d+\.\d+)_", effective_template)
        if m:
            parts.append(f"n{m.group(1)}")
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
