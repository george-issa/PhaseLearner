"""
Shared configuration: FSS reference values, paths, and physical constants.
"""
import numpy as np
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "NEW")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ── Physical parameters ───────────────────────────────────────────────────────
L = 12
SIZE = L * L  # 144 features per sample

ALPHAS = [0.240, 0.245, 0.250, 0.255]

# FSS reference values (w=0.5, from sohan_phase_diagram.pdf)
FSS_ALPHA = np.array([0.240, 0.245, 0.250, 0.255])
FSS_BETA_C = np.array([17.5, 14.3, 12.5, 12.3])

# Mapping from alpha to FSS beta_c for convenience
FSS_DICT = dict(zip(FSS_ALPHA, FSS_BETA_C))

# ── Device selection ──────────────────────────────────────────────────────────
# MPS can cause silent hangs with PyTorch 2.2; use CPU for reliability.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
