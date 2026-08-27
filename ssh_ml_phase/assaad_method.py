"""
Greitemann-Iqbal (Assaad) method for phase transition detection.

For each alpha, sweep guessed beta_c values. For each guess, train a CNN
to classify configurations from beta values on either side of the guess
(excluding `nskip` points around it). The peak test accuracy identifies
beta_c — the network distinguishes phases most easily when the guess is
correct and both training sides contain pure-phase data.

Relation to LBC
---------------
This is *not* Learning by Confusion.  Assaad trains on two pure β-windows on
either side of a guess and looks for a single accuracy *peak*; LBC labels the
whole dataset by each guess and looks for a W-curve middle peak.  Both share the
same diagnostic style here (data provenance + per-guess training summaries +
loss/accuracy/LR-vs-epoch plots).

Diagnostics
-----------
Per guess the training summary (losses, accuracies, learning rate, epochs) is
printed to the terminal, and the per-epoch history of the representative guess
(closest to the FSS β_c) is saved so ``plot.py`` can render
``training_curves.pdf`` (loss / accuracy / LR vs epoch) alongside the main
``accuracy.pdf`` (test accuracy vs guessed β_c).

Usage
-----
    python3.11 -m ssh_ml_phase.assaad_method                           # all alphas, all nskips
    python3.11 -m ssh_ml_phase.assaad_method --alphas 0.240            # single alpha
    python3.11 -m ssh_ml_phase.assaad_method --nskips 4 6              # subset of nskips
    python3.11 -m ssh_ml_phase.assaad_method --tag bmin8               # explicit output label
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from . import config as _cfg
from .config import DEVICE
from .data import (load_beta_sweep, ensure_npz_dir, result_filename,
                   lattice_snapshots, store_snapshots)

# Per-epoch metrics tracked during training (and plotted vs epoch).
_HIST_KEYS = ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]


class CNN_GI(nn.Module):
    """Small CNN for binary phase classification (Assaad/GI method)."""

    def __init__(self, L=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        flat_size = 16 * (L // 4) * (L // 4)
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_run(model, optimizer, scheduler, criterion,
                  train_loader, val_loader, nepochs, patience):
    """Train with early stopping on validation loss.

    Returns
    -------
    best_state : dict  — model ``state_dict`` at the lowest validation loss.
    history    : dict  — per-epoch lists for each of ``_HIST_KEYS``.
    best_epoch : int   — epoch index (0-based) of ``best_state``.
    """
    history = {k: [] for k in _HIST_KEYS}
    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    no_improve = 0

    for epoch in range(nepochs):
        model.train()
        train_loss, train_correct, n = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += (out.argmax(dim=1) == yb).sum().item()
            n += len(xb)
        train_loss /= n
        train_acc = train_correct / n

        model.eval()
        val_loss, val_correct, m = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
                val_correct += (out.argmax(dim=1) == yb).sum().item()
                m += len(xb)
        val_loss /= m
        val_acc = val_correct / m

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step(val_loss)

        if no_improve >= patience:
            break

    return best_state, history, best_epoch


def _guess_accuracy(X, betas, beta_left, beta_right, side, seed, *,
                    nepochs=60, batch_size=32, patience=15):
    """Train a CNN to separate the left vs right β-windows of one guess.

    Returns ``(test_acc, history, best_epoch)``.
    """
    X0 = X[np.isin(betas, beta_left)]
    X1 = X[np.isin(betas, beta_right)]

    # Balance classes
    n_min = min(len(X0), len(X1))
    X0, X1 = X0[:n_min], X1[:n_min]

    X_all = np.vstack([X0, X1])
    y_all = np.concatenate([np.zeros(n_min), np.ones(n_min)])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    Ntot = len(X_all)
    n_train = int(0.8 * Ntot)
    n_val = int(0.1 * Ntot)

    def to_X(a):
        return torch.tensor(a.reshape(-1, 1, side, side), dtype=torch.float32)

    X_tr = to_X(X_all[:n_train])
    y_tr = torch.tensor(y_all[:n_train], dtype=torch.long)
    X_va = to_X(X_all[n_train:n_train + n_val])
    y_va = torch.tensor(y_all[n_train:n_train + n_val], dtype=torch.long)
    X_te = to_X(X_all[n_train + n_val:])
    y_te = torch.tensor(y_all[n_train + n_val:], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size)

    torch.manual_seed(seed)
    model = CNN_GI(side).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_state, history, best_epoch = train_one_run(
        model, optimizer, scheduler, criterion,
        train_loader, val_loader, nepochs, patience)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te.to(DEVICE)).argmax(dim=1).cpu()
    test_acc = (preds == y_te).float().mean().item()
    return test_acc, history, best_epoch


def _stack_histories(runs):
    """Pack one guess's per-seed training curves into rectangular arrays.

    ``runs`` is a list of ``(history, best_epoch)``.  Returns a dict mapping
    each metric in ``_HIST_KEYS`` to an ``(n_seeds, max_epochs)`` array (shorter
    early-stopped runs are NaN-padded), plus ``best_epoch`` of shape
    ``(n_seeds,)``.
    """
    n_seeds = len(runs)
    max_ep = max(len(h["val_loss"]) for h, _ in runs)
    out = {}
    for k in _HIST_KEYS:
        arr = np.full((n_seeds, max_ep), np.nan, dtype=np.float32)
        for i, (h, _) in enumerate(runs):
            arr[i, :len(h[k])] = h[k]
        out[k] = arr
    out["best_epoch"] = np.array([be for _, be in runs])
    return out


def run_single(X, betas, nskip, fss_bc, side, n_seeds=1, base_seed=42,
               verbose=True, nepochs=60, batch_size=32, patience=15):
    """Run the Assaad sweep for one (alpha, nskip) on pre-loaded data.

    Each guess is trained over ``n_seeds`` random splits and the test accuracy
    is averaged (mean ± seed std).  Returns a dict with the guessed β_c grid,
    the mean/std test accuracies, the peak β_c, and the per-epoch training
    history of the representative guess (the one closest to the FSS β_c).
    """
    u_beta = np.unique(betas)
    half = nskip // 2
    NG = len(u_beta) - nskip
    trunc_beta = u_beta[half:-half]

    # Guess whose training history we keep for the training_curves diagnostic.
    rep_idx = int(np.argmin(np.abs(trunc_beta - fss_bc)))

    mean_acc = np.zeros(NG)
    std_acc = np.zeros(NG)
    rep_runs = None

    for iL in range(NG):
        beta_left = u_beta[iL: iL + half]
        beta_right = u_beta[iL + half + 1: iL + nskip + 1]

        accs, runs = [], []
        for s in range(n_seeds):
            seed = base_seed + iL * 100 + s
            acc, history, best_epoch = _guess_accuracy(
                X, betas, beta_left, beta_right, side, seed=seed,
                nepochs=nepochs, batch_size=batch_size, patience=patience)
            accs.append(acc)
            runs.append((history, best_epoch))
            if verbose:
                be = best_epoch
                seed_tag = f" seed {s}:" if n_seeds > 1 else ":"
                print(f"  nskip={nskip} guess={trunc_beta[iL]:5.2f}{seed_tag} "
                      f"epochs={len(history['val_loss'])} (best@{be})  "
                      f"loss tr/va={history['train_loss'][be]:.3f}/"
                      f"{history['val_loss'][be]:.3f}  "
                      f"acc tr/va={history['train_acc'][be]:.3f}/"
                      f"{history['val_acc'][be]:.3f}  "
                      f"lr={history['lr'][be]:.1e}  test_acc={acc:.3f}")

        mean_acc[iL] = float(np.mean(accs))
        std_acc[iL] = float(np.std(accs))
        if iL == rep_idx:
            rep_runs = runs
        if verbose and n_seeds > 1:
            print(f"  nskip={nskip} guess={trunc_beta[iL]:5.2f}  ->  "
                  f"mean test acc = {mean_acc[iL]:.3f} ± {std_acc[iL]:.3f}")

    return {
        "betas":      trunc_beta,
        "accs":       mean_acc,
        "accs_std":   std_acc,
        "beta_c":     float(trunc_beta[int(np.argmax(mean_acc))]),
        "hist_beta_c_prime": float(trunc_beta[rep_idx]),
        "hist":       _stack_histories(rep_runs) if rep_runs else None,
    }


def run_all(alphas=None, nskip_values=None, tag=None, n_seeds=1,
            beta_min=None, beta_max=None, stride=None, file_template=None):
    """
    Run Assaad method for all alphas and nskip values.
    Saves to results/npz/assaad_results.npz.

    ``n_seeds`` retrains each guess over that many random splits and reports the
    mean ± seed-std test accuracy (1 → single-seed, no error bars).
    """
    if alphas is None:
        alphas = _cfg.ALPHAS
    if nskip_values is None:
        nskip_values = [4, 6, 8]

    results_dir = ensure_npz_dir()
    all_results = {}
    snapshots = {}

    for alpha in alphas:
        print(f"\n[Assaad] alpha={alpha:.3f} ...")
        X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                                   stride=stride, file_template=file_template)
        u_beta = np.unique(betas)
        fss_bc = _cfg.FSS_DICT.get(alpha, float(np.mean(u_beta)))

        # Data provenance — shape, dtype, and the β grid being classified.
        print(f"  X: shape={X.shape}, dtype={X.dtype}  |  "
              f"betas: shape={betas.shape}, dtype={betas.dtype}")
        print(f"  unique betas ({len(u_beta)}): "
              f"{np.array2string(u_beta, precision=2, separator=', ')}")

        # The CNN reshapes each feature vector onto the L×L lattice.
        n_features = X.shape[1]
        side = int(round(n_features ** 0.5))
        if side * side != n_features:
            raise ValueError(
                f"Assaad's CNN needs square-lattice features (n_features = L²), "
                f"but got n_features={n_features} (√≈{n_features ** 0.5:.2f}). Use "
                f"a raw L² observable, or pick the pca/prediction/mean method."
            )

        snapshots[alpha] = lattice_snapshots(X, betas)
        all_results[alpha] = {}
        for nskip in nskip_values:
            res = run_single(X, betas, nskip, fss_bc, side,
                             n_seeds=n_seeds, verbose=True)
            all_results[alpha][nskip] = res
            print(f"  nskip={nskip}: beta_c={res['beta_c']:.2f}  "
                  f"peak_acc={np.max(res['accs']):.4f}  FSS={fss_bc:.2f}")

    # Save
    save_dict = {}
    for alpha in alphas:
        akey = f"a{alpha:.4f}"
        inlat_betas, inlat_grids = snapshots[alpha]
        store_snapshots(save_dict, akey,
                        {"inlat_betas": inlat_betas, "inlat_grids": inlat_grids})
        for nskip in nskip_values:
            r = all_results[alpha][nskip]
            key = f"{akey}_ns{nskip}"
            save_dict[f"{key}_betas"] = r["betas"]
            save_dict[f"{key}_accs"] = r["accs"]
            save_dict[f"{key}_accs_std"] = r["accs_std"]
            save_dict[f"{key}_beta_c"] = np.array([r["beta_c"]])
            # Per-epoch training history of the representative guess (per seed).
            if r["hist"] is not None:
                save_dict[f"{key}_hist_beta_c_prime"] = np.array([r["hist_beta_c_prime"]])
                for k in _HIST_KEYS:
                    save_dict[f"{key}_hist_{k}"] = r["hist"][k]
                save_dict[f"{key}_hist_best_epoch"] = r["hist"]["best_epoch"]

    fname = result_filename("assaad", tag)
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Assaad (GI) method for beta_c")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Named model preset (sets alphas, file-template, FSS). "
                             "Explicit --alphas / --file-template override the preset.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--nskips", nargs="+", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Random splits averaged per guess (1 → single-seed)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Label appended to output filename")
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None,
                        help="Use every N-th row (subsample dense datasets)")
    parser.add_argument("--file-template", type=str, default=None,
                        help="Filename template with {alpha:.4f} placeholder")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing input CSV files")
    args = parser.parse_args()

    if args.model is not None:
        _cfg.apply_model(args.model)
    if args.data_dir is not None:
        _cfg.DATA_DIR = args.data_dir

    alphas        = args.alphas        or _cfg.ALPHAS
    file_template = args.file_template or _cfg.FILE_TEMPLATE

    run_all(alphas=alphas, nskip_values=args.nskips, tag=args.tag,
            n_seeds=args.n_seeds,
            beta_min=args.beta_min, beta_max=args.beta_max, stride=args.stride,
            file_template=file_template)


if __name__ == "__main__":
    main()
