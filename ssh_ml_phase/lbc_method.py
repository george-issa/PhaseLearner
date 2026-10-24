"""
Learning by Confusion (LBC) — van Nieuwenburg, Liu & Huber, Nat. Phys. 13, 435 (2017).

Idea
----
We don't know β_c, so we guess.  For each trial value β_c', we *pretend* the
transition is there: label every configuration y = 1[β > β_c'] and train a
binary classifier.  The held-out accuracy as a function of the guess traces the
characteristic **W-shape**:

  * At the two extremes the guess puts (almost) all data in one class, so the
    classifier is trivially right → accuracy ≈ 1.
  * For a wrong interior guess the labels split a single physical phase, so the
    classifier is "confused" → accuracy dips.
  * At the *true* β_c the two labels coincide with the two real phases, which
    the classifier separates easily → accuracy peaks again.

β_c is read off as the interior (middle) peak of the W.

Classifier
----------
A small CNN (`SimpleCNN`) trained with early stopping — same architecture and
training structure as the standalone ``scripts/lbc_train.py``: one conv block + an MLP
head with a single sigmoid output, BCE loss, an Adam optimiser, and a
``ReduceLROnPlateau`` schedule monitored on a validation split.  The feature
vector is reshaped onto the L×L lattice, so this method needs square-lattice
inputs (``n_features = L²``); transformed / radial observables are not square
and should use the PCA / prediction / mean methods instead.

Diagnostics
-----------
Per (candidate, seed) the training summary (losses, accuracies, learning rate,
epochs) is printed to the terminal, and the per-epoch history of the
representative candidate (the W peak) is saved so ``plot.py`` can render
``training_curves.pdf`` (loss / accuracy / LR vs epoch).

Relation to the other methods
-----------------------------
`assaad_method.py` is a *different* supervised scheme (Greitemann–Iqbal): it
trains on two pure β-windows on either side of a guess and looks for a single
accuracy peak.  LBC instead labels the *whole* dataset by each guess and looks
for the W-curve's middle peak.

Usage
-----
    python3.11 -m ssh_ml_phase.lbc_method --model attractive_hubbard
    python3.11 -m ssh_ml_phase.lbc_method --alphas 0.240 --n-seeds 5
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from . import config as _cfg
from .config import DEVICE
from .data import (load_beta_sweep, ensure_npz_dir, make_tag, result_filename,
                   lattice_snapshots, store_snapshots)

# Per-epoch metrics tracked during training (and plotted vs epoch).
_HIST_KEYS = ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]


# ----------------------------------------------------------------------
# Model — mirrors lbc_train.SimpleCNN
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Small CNN: one conv block + MLP head, single-logit binary output."""

    def __init__(self, L, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear((L // 2) ** 2 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.flatten(1)
        return self.classifier(out).squeeze(-1)


# ----------------------------------------------------------------------
# Training loop — mirrors lbc_train.train_one_run (best state kept in memory)
# ----------------------------------------------------------------------
def train_one_run(model, optimizer, scheduler, criterion,
                  train_loader, val_loader, num_epochs, patience_stop):
    """Train with early stopping on validation loss.

    Returns
    -------
    best_state : dict   — model ``state_dict`` at the lowest validation loss.
    history    : dict   — per-epoch lists for each of ``_HIST_KEYS``.
    best_epoch : int    — epoch index (0-based) of ``best_state``.
    """
    history = {k: [] for k in _HIST_KEYS}
    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, n = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += ((out >= 0.5).float() == yb).float().sum().item()
            n += xb.size(0)
        train_loss /= n
        train_acc = train_correct / n

        model.eval()
        val_loss, val_correct, m = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                val_correct += ((out >= 0.5).float() == yb).float().sum().item()
                m += xb.size(0)
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

        if no_improve >= patience_stop:
            break

    return best_state, history, best_epoch


# ----------------------------------------------------------------------
# One trained classifier for one guess β_c'  (one seed)
# ----------------------------------------------------------------------
def _candidate_accuracy(X, betas, bc, side, seed, *,
                        num_epochs=150, batch_size=32, dropout=0.2,
                        lr=1e-3, weight_decay=1e-2,
                        patience_lr=10, patience_stop=20):
    """Train a CNN on labels ``y = 1[β > bc]`` and return held-out test accuracy.

    Stable train/val/test split (test 0.1, val 0.15 of the rest); features are
    z-scored using *training* statistics only (no test leakage).

    Returns ``(test_acc, history, best_epoch)``.
    """
    y = (betas > bc).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=seed, shuffle=True)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed, shuffle=True)

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8

    def to_tensor(Xa):
        return torch.tensor(((Xa - mean) / std).reshape(-1, 1, side, side),
                            dtype=torch.float32)

    X_tr_t, X_va_t, X_te_t = to_tensor(X_tr), to_tensor(X_va), to_tensor(X_te)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=batch_size)

    torch.manual_seed(seed)
    model = SimpleCNN(L=side, dropout=dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=patience_lr)
    criterion = nn.BCELoss()

    best_state, history, best_epoch = train_one_run(
        model, optimizer, scheduler, criterion,
        train_loader, val_loader, num_epochs, patience_stop)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = (model(X_te_t.to(DEVICE)) >= 0.5).float()
    test_acc = (preds == y_te_t).float().mean().item()
    return test_acc, history, best_epoch


def candidate_grid(betas):
    """Trial β_c' values for the confusion sweep.

    Midpoints between consecutive unique β (each cleanly splits the β-grid),
    bracketed by one point just outside each end.  At those two outer points
    every sample falls on one side, giving the trivially-separable anchors that
    form the high arms of the W-curve.
    """
    grid = np.unique(betas)
    mids = 0.5 * (grid[:-1] + grid[1:])
    half_step = 0.5 * (grid[1] - grid[0])
    return np.concatenate([[grid[0] - half_step], mids, [grid[-1] + half_step]])


def confusion_sweep(X, betas, side, n_seeds=3, verbose=True, **train_kw):
    """Run the LBC sweep over all candidate β_c' values.

    For each candidate the CNN is retrained from scratch on the labels
    ``y = 1[β > β_c']`` and the held-out accuracy is averaged over ``n_seeds``
    random splits.  An outer-anchor candidate (one class empty) is trivially
    separable, so its accuracy is the majority-class fraction (= 1.0) computed
    directly instead of training a degenerate network.

    Returns
    -------
    candidates, mean_acc, std_acc : ndarray, each shape (n_candidates,)
    histories : list, length n_candidates
        Each entry is a list of ``(history, best_epoch)`` per seed (empty for a
        trivial anchor).  Used to plot loss/accuracy/LR vs epoch.
    """
    candidates = candidate_grid(betas)
    mean_acc = np.zeros(len(candidates))
    std_acc = np.zeros(len(candidates))
    histories = []

    for i, bc in enumerate(candidates):
        y = (betas > bc)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        if n_pos == 0 or n_neg == 0:
            # Outer anchor: one class is empty → trivially separable.
            mean_acc[i] = max(n_pos, n_neg) / len(y)
            std_acc[i] = 0.0
            histories.append([])
            if verbose:
                print(f"  beta_c'={bc:6.2f}  (anchor: one class empty)  "
                      f"acc={mean_acc[i]:.3f}")
            continue

        accs, runs = [], []
        for s in range(n_seeds):
            seed = 2000 + s
            acc, history, best_epoch = _candidate_accuracy(
                X, betas, bc, side, seed=seed, **train_kw)
            accs.append(acc)
            runs.append((history, best_epoch))
            if verbose:
                be = best_epoch
                print(f"  beta_c'={bc:6.2f}  seed {s}: "
                      f"epochs={len(history['val_loss'])} (best@{be})  "
                      f"loss tr/va={history['train_loss'][be]:.3f}/"
                      f"{history['val_loss'][be]:.3f}  "
                      f"acc tr/va={history['train_acc'][be]:.3f}/"
                      f"{history['val_acc'][be]:.3f}  "
                      f"lr={history['lr'][be]:.1e}  test_acc={acc:.3f}")

        mean_acc[i] = float(np.mean(accs))
        std_acc[i] = float(np.std(accs))
        histories.append(runs)
        if verbose:
            print(f"  beta_c'={bc:6.2f}  ->  mean test acc = "
                  f"{mean_acc[i]:.3f} ± {std_acc[i]:.3f}")

    return candidates, mean_acc, std_acc, histories


def w_middle_peak(candidates, mean_acc):
    """β_c = the middle peak of the LBC W-curve.

    Returns the candidate at the *highest strict local maximum* in the interior
    — the bump between the two confusion dips, which is what LBC identifies as
    β_c.  The two trivially-perfect outer arms can therefore never win.  If the
    curve has no interior local maximum (no clean W), falls back to the highest
    interior candidate.
    """
    local_maxima = [
        i for i in range(1, len(candidates) - 1)
        if mean_acc[i] > mean_acc[i - 1] and mean_acc[i] > mean_acc[i + 1]
    ]
    if local_maxima:
        idx = max(local_maxima, key=lambda i: mean_acc[i])
    else:
        idx = int(np.argmax(mean_acc[1:-1])) + 1
    return float(candidates[idx])


def _stack_histories(runs):
    """Pack one candidate's per-seed training curves into rectangular arrays.

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


def compute_lbc(alpha, n_seeds=1, beta_min=None, beta_max=None, stride=None,
                file_template=None, **train_kw):
    """Run LBC for one alpha and extract β_c from the W-curve middle peak."""
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    u_beta = np.unique(betas)
    fss_bc = _cfg.FSS_DICT.get(alpha, float(np.mean(u_beta)))

    # Data provenance — shape, dtype, and the β grid actually being classified.
    print(f"  X: shape={X.shape}, dtype={X.dtype}  |  "
          f"betas: shape={betas.shape}, dtype={betas.dtype}")
    print(f"  unique betas ({len(u_beta)}): "
          f"{np.array2string(u_beta, precision=2, separator=', ')}")

    # The CNN reshapes each feature vector onto the L×L lattice.
    n_features = X.shape[1]
    side = int(round(n_features ** 0.5))
    if side * side != n_features:
        raise ValueError(
            f"LBC's CNN needs square-lattice features (n_features = L²), but got "
            f"n_features={n_features} (√≈{n_features ** 0.5:.2f}). Use a raw L² "
            f"observable, or pick the pca/prediction/mean method for this input."
        )

    candidates, mean_acc, std_acc, histories = confusion_sweep(
        X, betas, side, n_seeds=n_seeds, **train_kw)
    beta_c = w_middle_peak(candidates, mean_acc)

    # Per-epoch training curves of the representative candidate (the W peak),
    # for the training_curves.pdf diagnostic.
    peak_idx = int(np.argmin(np.abs(candidates - beta_c)))
    peak_runs = histories[peak_idx]
    hist = _stack_histories(peak_runs) if peak_runs else None

    inlat_betas, inlat_grids = lattice_snapshots(X, betas)

    return {
        "u_beta":      u_beta,
        "candidates":  candidates,
        "mean_acc":    mean_acc,
        "std_acc":     std_acc,
        "beta_c":      beta_c,
        "fss_bc":      fss_bc,
        "hist_beta_c_prime": float(candidates[peak_idx]),
        "hist":        hist,
        "inlat_betas": inlat_betas,
        "inlat_grids": inlat_grids,
        "n_features":  n_features,
        "n_samples":   len(X),
    }


def run_all(alphas=None, n_seeds=1, beta_min=None, beta_max=None, stride=None,
            file_template=None, tag=None):
    """Run LBC for all alphas and save to results/npz/.

    β_c array layout: ``[w_peak]`` — single estimate from the W middle peak.
    """
    if alphas is None:
        alphas = _cfg.ALPHAS

    results_dir = ensure_npz_dir()
    all_results = {}

    for alpha in alphas:
        print(f"\n[LBC] alpha={alpha:.3f} ...")
        res = compute_lbc(
            alpha, n_seeds=n_seeds,
            beta_min=beta_min, beta_max=beta_max, stride=stride,
            file_template=file_template,
        )
        all_results[alpha] = res

        fss = _cfg.FSS_DICT.get(alpha, np.nan)
        print(f"  beta_c (W peak) = {res['beta_c']:.2f}  FSS={fss:.2f}  "
              f"(n_features={res['n_features']}, n_samples={res['n_samples']})")

    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]     = r["u_beta"]
        save_dict[f"{key}_candidates"] = r["candidates"]
        save_dict[f"{key}_mean_acc"]   = r["mean_acc"]
        save_dict[f"{key}_std_acc"]    = r["std_acc"]
        save_dict[f"{key}_beta_c"]     = np.array([r["beta_c"]])
        store_snapshots(save_dict, key, r)
        # Per-epoch training history of the representative (W-peak) candidate.
        if r["hist"] is not None:
            save_dict[f"{key}_hist_beta_c_prime"] = np.array([r["hist_beta_c_prime"]])
            for k in _HIST_KEYS:
                save_dict[f"{key}_hist_{k}"] = r["hist"][k]
            save_dict[f"{key}_hist_best_epoch"] = r["hist"]["best_epoch"]

    fname = result_filename(
        "lbc",
        make_tag(beta_min, file_template, tag,
                 beta_max=beta_max, stride=stride),
    )
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Learning by Confusion β_c extraction (van Nieuwenburg 2017)"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_cfg.MODEL_CONFIGS),
                        help="Named model preset (sets alphas, file-template, FSS)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--n-seeds", type=int, default=1)
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
