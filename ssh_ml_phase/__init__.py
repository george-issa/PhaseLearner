"""
ssh_ml_phase: ML methods for detecting phase transitions in the optical SSH model.

Submodules:
    config          - FSS reference values, paths, constants
    data            - Data loading utilities
    pca_method      - PCA-based beta_c extraction
    umap_method     - UMAP-based beta_c extraction (KMeans + GMM)
    assaad_method   - Greitemann-Iqbal (Assaad) CNN method
    lbc_method      - Learning by Confusion (van Nieuwenburg 2017)
    autoencoder     - Autoencoder anomaly detection
"""

from . import config
from . import data
