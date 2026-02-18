"""
Data loading utilities for augmented BondBond CSV files.
"""
import numpy as np
import os
from .config import DATA_DIR, SIZE, RESULTS_DIR


def load_data(alpha, data_dir=None):
    """
    Load augmented BondBond012345 momentum data for a given alpha.

    Parameters
    ----------
    alpha : float
        Coupling constant (e.g. 0.240, 0.245, 0.250, 0.255).
    data_dir : str, optional
        Override default data directory.

    Returns
    -------
    X : ndarray, shape (N, 144)
        Feature matrix (12x12 lattice configurations, flattened).
    betas : ndarray, shape (N,)
        Inverse temperature for each sample.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    fname = os.path.join(
        data_dir,
        f"BondBond012345_momentum_L12_w0.5_m0.0_a{alpha:.4f}-1.csv",
    )
    data = np.loadtxt(fname, delimiter=",")
    X = data[:, :SIZE]
    betas = data[:, -1]
    return X, betas


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
