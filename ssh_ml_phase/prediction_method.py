"""
Prediction-based β_c extraction (Schäfer-Lörch / Greitemann-Iqbal).

Strategy
--------
Train an MLP to regress the inverse temperature β from each configuration.
After training, evaluate ⟨β_pred⟩(β_true):
  * away from a transition, predicted β is roughly *linear* in true β
    → constant derivative;
  * across a transition, the NN cannot tell configurations within a single
    phase apart, so predictions saturate inside each phase and rise sharply
    through β_c → derivative *peaks* at β_c.

β_c extraction
--------------
**Raw-derivative peak** — β_c = argmax_β |d⟨β_pred⟩/dβ| on the unsmoothed
per-β mean, with the first / last few β-points excluded to avoid
finite-difference edge artifacts.  No Gaussian smoothing anywhere in the
pipeline (per user 2026-05-20).

Reference
---------
Dawid et al., "Modern applications of machine learning in quantum sciences,"
arXiv:2204.04198v4 §3.4.3 (Fig. 3.7); methodology of Schäfer & Lörch and
Greitemann–Iqbal.

Usage
-----
    python3.11 -m ssh_ml_phase.prediction_method                         # default config
    python3.11 -m ssh_ml_phase.prediction_method --model attractive_hubbard
    python3.11 -m ssh_ml_phase.prediction_method --n-seeds 5
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from . import config as _cfg
from .config import DEVICE
from .data import (load_beta_sweep, ensure_npz_dir, make_tag, result_filename,
                   lattice_snapshots, store_snapshots)


class MLPRegressor(nn.Module):
    """Fully-connected regressor `n_features → 512 → 256 → 128 → 64 → 1`
    with ReLU + dropout between each hidden layer.

    Depth and width follow the spirit of the references that introduced the
    prediction-based method:
      * Schäfer & Lörch 2019 (arXiv:1812.00895) — 3 conv + 3 FC, ReLU.
      * Greplova et al. 2020 (arXiv:1910.10124) — 2 conv + 2 dense (300, 100),
        dropout 0.15.
      * Arnold et al. 2021 (arXiv:2010.04730) — wide FC stack
        (2048→1024→512→512→256→2), Adam, dropout-regularised.

    For k-space DQMC features (Hubbard P_s(q), n_features ≈ 64) convolutions
    aren't physically motivated, so we drop them and use a deeper FC stack
    plus dropout to give the model real capacity without overfitting.
    """

    def __init__(self, n_features, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _single_run(X_norm, y_norm, n_features, nepochs, batch_size, lr, seed):
    """Train one MLPRegressor.

    Returns
    -------
    dict with keys:
        preds          : (N,) z-scored per-sample predictions of β
        train_losses   : (n_epochs_run,) per-epoch training MSE (z-units)
        val_losses     : (n_epochs_run,) per-epoch validation MSE (z-units)
        best_epoch     : int  — epoch index of the best val loss
        best_val_loss  : float — best validation MSE (z-units)
        best_train_loss: float — training MSE at the best epoch
        n_epochs_run   : int  — total epochs before early-stop
    """
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    n_val = max(int(0.1 * len(X_t)), 1)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X_t), generator=gen)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    n_train = len(train_idx)

    train_loader = DataLoader(
        TensorDataset(X_t[train_idx], y_t[train_idx]),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_t[val_idx], y_t[val_idx]),
        batch_size=batch_size,
    )

    torch.manual_seed(seed)
    model = MLPRegressor(n_features=n_features).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Patience 30 + early-stop patience 50 = roughly Arnold et al.'s tuning
    # (their LR scheduler uses patience 50 inside a ~770-1576 epoch budget).
    # Factor 0.5 matches Schäfer & Lörch's "halve LR" recipe.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses   = []
    lr_curve     = []
    best_val_loss   = float("inf")
    best_train_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(nepochs):
        model.train()
        epoch_train_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_sum += loss.item() * len(xb)
        epoch_train_loss = epoch_train_sum / n_train

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_sum += criterion(pred, yb).item() * len(xb)
        val_loss = val_sum / n_val

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        lr_curve.append(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_train_loss = epoch_train_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 50:   # Arnold-style early-stop patience
            break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    full_loader = DataLoader(TensorDataset(X_t), batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in full_loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return {
        "preds":           np.concatenate(preds),
        "train_losses":    np.array(train_losses,  dtype=np.float32),
        "val_losses":      np.array(val_losses,    dtype=np.float32),
        "lr_curve":        np.array(lr_curve,      dtype=np.float32),
        "best_epoch":      int(best_epoch),
        "best_val_loss":   float(best_val_loss),
        "best_train_loss": float(best_train_loss),
        "n_epochs_run":    len(train_losses),
    }


def train_and_evaluate(alpha, nepochs=500, batch_size=64, lr=3e-4,
                       n_seeds=5, verbose=True,
                       beta_min=None, beta_max=None, stride=None,
                       file_template=None):
    """Train MLP regressor (multi-seed average) and extract β_c."""
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    inlat_betas, inlat_grids = lattice_snapshots(X, betas)
    u_beta = np.unique(betas)
    n_features = X.shape[1]
    fss_bc = _cfg.FSS_DICT.get(alpha, float(np.mean(u_beta)))

    # Standardize features (per-feature z-score)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    # Standardize labels (β z-score) → un-standardize the prediction afterwards
    y_mean = float(betas.mean())
    y_std  = float(betas.std() + 1e-10)
    y_norm = (betas - y_mean) / y_std

    if verbose:
        print(f"  n_features={n_features}, n_beta={len(u_beta)}, "
              f"N={len(X)}, {n_seeds} seeds...")

    all_seed_preds = []
    all_seed_diag  = []  # train/val loss curves + early-stop metadata
    for s in range(n_seeds):
        seed = 42 + s * 100
        r = _single_run(
            X_norm, y_norm, n_features, nepochs, batch_size, lr, seed,
        )
        pred = r["preds"] * y_std + y_mean
        all_seed_preds.append(pred)
        all_seed_diag.append(r)
        if verbose:
            mean_pred_s = np.array([np.mean(pred[betas == b]) for b in u_beta])
            d_s = np.diff(mean_pred_s) / np.diff(u_beta)
            bc_s = float(0.5 * (u_beta[np.argmax(np.abs(d_s))] +
                                u_beta[np.argmax(np.abs(d_s)) + 1]))
            print(f"    seed {s}: deriv peak at beta={bc_s:.2f}  "
                  f"best_val={r['best_val_loss']:.4f} (epoch {r['best_epoch']}, "
                  f"ran {r['n_epochs_run']} epochs)")

    avg_pred = np.mean(all_seed_preds, axis=0)

    # ── Convergence diagnostics ──────────────────────────────────────────────
    # Pad per-seed loss curves into rectangular arrays (NaN where early-stop
    # ended training early) for npz save.  Units: MSE in z-scored β space, so
    # a value of 1.0 ≈ "predict the mean," 0.0 ≈ "perfect."  Var(β_true) in
    # z-units is 1.0 by construction → val_loss / 1.0 = unexplained variance,
    # so 1 − val_loss ≈ R² on the validation split.
    max_epochs = max(d["n_epochs_run"] for d in all_seed_diag)
    train_curves = np.full((n_seeds, max_epochs), np.nan, dtype=np.float32)
    val_curves   = np.full((n_seeds, max_epochs), np.nan, dtype=np.float32)
    lr_curves    = np.full((n_seeds, max_epochs), np.nan, dtype=np.float32)
    for i, d in enumerate(all_seed_diag):
        n = d["n_epochs_run"]
        train_curves[i, :n] = d["train_losses"]
        val_curves[i, :n]   = d["val_losses"]
        lr_curves[i, :n]    = d["lr_curve"]
    best_val_per_seed   = np.array([d["best_val_loss"]   for d in all_seed_diag])
    best_train_per_seed = np.array([d["best_train_loss"] for d in all_seed_diag])
    best_epoch_per_seed = np.array([d["best_epoch"]      for d in all_seed_diag])
    n_epochs_per_seed   = np.array([d["n_epochs_run"]    for d in all_seed_diag])

    # Per-β statistics
    mean_pred = np.array([np.mean(avg_pred[betas == b]) for b in u_beta])
    std_pred  = np.array([np.std(avg_pred[betas == b])  for b in u_beta])

    # Derivative curve at β midpoints (raw, no smoothing)
    beta_mid = 0.5 * (u_beta[:-1] + u_beta[1:])
    d_pred   = np.diff(mean_pred) / np.diff(u_beta)

    # Edge-skip avoids finite-difference artifacts at the β-grid boundaries
    n_mid = len(beta_mid)
    skip  = max(2, n_mid // 12)
    sl    = slice(skip, n_mid - skip)

    bc_deriv = float(beta_mid[sl][np.argmax(np.abs(d_pred)[sl])])

    return {
        "u_beta":          u_beta,
        "mean_pred":       mean_pred,
        "std_pred":        std_pred,
        "betas":           betas,
        "all_preds":       avg_pred,
        "inlat_betas":     inlat_betas,
        "inlat_grids":     inlat_grids,
        # β_c estimate (raw argmax of |d⟨β_pred⟩/dβ|)
        "beta_c_deriv":    bc_deriv,
        # curve
        "deriv_curve":     (beta_mid, d_pred),
        "fss_bc":          fss_bc,
        # training diagnostics (n_seeds, max_epochs)
        "train_curves":          train_curves,
        "val_curves":            val_curves,
        "lr_curves":             lr_curves,
        "best_val_per_seed":     best_val_per_seed,
        "best_train_per_seed":   best_train_per_seed,
        "best_epoch_per_seed":   best_epoch_per_seed,
        "n_epochs_per_seed":     n_epochs_per_seed,
    }


def run_all(alphas=None, n_seeds=5, beta_min=None, beta_max=None,
            stride=None, file_template=None, tag=None):
    """Run prediction-based analysis for all alphas; save to results/npz/.

    β_c array layout: ``[deriv]`` — single raw-argmax estimate.
    """
    if alphas is None:
        alphas = _cfg.ALPHAS

    results_dir = ensure_npz_dir()
    all_results = {}

    for alpha in alphas:
        print(f"\n[Prediction] alpha={alpha:.3f} ...")
        res = train_and_evaluate(
            alpha, n_seeds=n_seeds,
            beta_min=beta_min, beta_max=beta_max, stride=stride,
            file_template=file_template,
        )
        all_results[alpha] = res

        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print(f"  beta_c: deriv={res['beta_c_deriv']:.2f}  FSS={fss:.2f}")

    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]    = r["u_beta"]
        save_dict[f"{key}_mean_pred"] = r["mean_pred"]
        save_dict[f"{key}_std_pred"]  = r["std_pred"]
        store_snapshots(save_dict, key, r)
        # β_c: [deriv]  (raw argmax of |d⟨β_pred⟩/dβ|)
        save_dict[f"{key}_beta_c"]    = np.array([r["beta_c_deriv"]])
        save_dict[f"{key}_deriv_beta"] = r["deriv_curve"][0]
        save_dict[f"{key}_deriv_vals"] = r["deriv_curve"][1]
        # Scatter sample for plots
        rng = np.random.default_rng(42)
        idx = rng.choice(
            len(r["all_preds"]),
            min(5000, len(r["all_preds"])), replace=False,
        )
        save_dict[f"{key}_pred_sample"]  = r["all_preds"][idx]
        save_dict[f"{key}_betas_sample"] = r["betas"][idx]
        # Training diagnostics
        save_dict[f"{key}_train_curves"]        = r["train_curves"]
        save_dict[f"{key}_val_curves"]          = r["val_curves"]
        save_dict[f"{key}_lr_curves"]           = r["lr_curves"]
        save_dict[f"{key}_best_val_per_seed"]   = r["best_val_per_seed"]
        save_dict[f"{key}_best_train_per_seed"] = r["best_train_per_seed"]
        save_dict[f"{key}_best_epoch_per_seed"] = r["best_epoch_per_seed"]
        save_dict[f"{key}_n_epochs_per_seed"]   = r["n_epochs_per_seed"]

    fname = result_filename(
        "predict",
        make_tag(beta_min, file_template, tag,
                 beta_max=beta_max, stride=stride),
    )
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Prediction-based β_c extraction")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Named model preset (sets alphas, file-template, FSS)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--file-template", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if args.model is not None:
        _cfg.apply_model(args.model)
    if args.data_dir is not None:
        _cfg.DATA_DIR = args.data_dir

    alphas        = args.alphas        or _cfg.ALPHAS
    file_template = args.file_template or _cfg.FILE_TEMPLATE

    run_all(alphas=alphas, n_seeds=args.n_seeds,
            beta_min=args.beta_min, beta_max=args.beta_max, stride=args.stride,
            file_template=file_template, tag=args.tag)


if __name__ == "__main__":
    main()
