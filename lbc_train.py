"""Learning-by-confusion (LBC) training for the bond-Holstein model.

Trains a small CNN on DQMC-generated electron-density snapshots labeled by a
sweep parameter (typically the inverse temperature beta or the dimensionless
inverse coupling 1/lambda).  For every guess of the critical sweep value, the
classifier is re-trained from scratch on snapshots labeled by 'sweep < tc',
and the held-out accuracy is recorded.  Plotting the resulting accuracies as a
function of tc yields the characteristic LBC W-shape, whose interior minimum
identifies the true critical value.

Reference: van Nieuwenburg, Liu and Huber, Nat. Phys. 13, 435 (2017).

Inputs
------
A single CSV with shape (N_samples, L*L + 1) where the first L*L columns are
one electron-density snapshot and the last column is the sweep parameter.

Outputs
-------
A directory `<output_folder>/` containing per-tc training artefacts and a
top-level `accs_w<w>_a<alpha>_L<L>_seed<seed>-<sID>.csv` file holding one
held-out accuracy per guess of the critical sweep value.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
def augment_with_flips(data: np.ndarray, L: int) -> np.ndarray:
    """Augment by four distinct square-lattice symmetries.  Last column is the label.

    Assumes the L x L lattice is symmetric under the C4v point group (true for the
    isotropic square lattice); if the model is anisotropic, drop the transpose.
    """
    rows, cols = data.shape
    cfgs = data[:, :-1].reshape(rows, L, L)
    labels = data[:, -1:]

    flips = [
        np.flip(cfgs, axis=1),          # reflect vertically
        np.flip(cfgs, axis=2),          # reflect horizontally
        np.flip(cfgs, axis=(1, 2)),     # 180-degree rotation
        np.swapaxes(cfgs, 1, 2),        # reflect across the main diagonal
    ]
    flat_flips = [np.concatenate([f.reshape(rows, -1), labels], axis=1) for f in flips]
    augmented = np.vstack([data, *flat_flips])
    augmented = augmented[augmented[:, -1].argsort()]
    return augmented


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Small CNN: one conv block + MLP head, single-logit binary output."""

    def __init__(self, L: int, dropout: float = 0.2):
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
# Training loop
# ----------------------------------------------------------------------
def train_one_run(model, optimizer, scheduler, criterion, train_loader, val_loader,
                  num_epochs: int, patience_stop: int, ckpt_path: Path):
    """Train with early stopping; save best model to ckpt_path. Returns history dict."""
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_acc += ((out >= 0.5).float() == yb).float().sum().item()
            n += xb.size(0)
        train_loss /= n
        train_acc /= n

        model.eval()
        val_loss, val_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_acc += ((out >= 0.5).float() == yb).float().sum().item()
                n += xb.size(0)
        val_loss /= n
        val_acc /= n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        scheduler.step(val_loss)

        if no_improve >= patience_stop:
            break

    return history


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------
def run_lbc(
    *, input_csv: Path, output_dir: Path,
    w: float, alpha: float, L: int, sID: int, seed: int,
    tc_min: float, tc_max: float, num_tc: int,
    batch_size: int = 32, num_epochs: int = 150, dropout: float = 0.2,
    initial_lr: float = 1e-3, weight_decay: float = 1e-2,
    patience_lr: int = 10, patience_stop: int = 20,
) -> Path:
    """Run an LBC sweep.  Returns the path to the accs CSV that was written."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch {torch.__version__}; loading {input_csv}")
    data = np.loadtxt(input_csv, delimiter=",")
    augmented = augment_with_flips(data, L)
    np.savetxt(output_dir / f"augmented_data_w{w:.2f}_a{alpha:.4f}_L{L}-{sID}.csv",
               augmented, delimiter=",")

    X_all = augmented[:, :-1]
    sweep = augmented[:, -1]
    print(f"  X.shape = {X_all.shape}, sweep in [{sweep.min():.3f}, {sweep.max():.3f}]")

    tc_values = np.linspace(tc_min, tc_max, num_tc)
    test_accuracies = []

    for tc in tc_values:
        tc_dir = output_dir / f"tc{tc:.4f}"
        tc_dir.mkdir(exist_ok=True)

        # Stable train/val/test split, independent of tc, so curves are comparable.
        X_tr, X_te, t_tr, t_te = train_test_split(X_all, sweep, test_size=0.1,
                                                  random_state=1, shuffle=True)
        X_tr, X_va, t_tr, t_va = train_test_split(X_tr, t_tr, test_size=0.15,
                                                  random_state=1, shuffle=True)

        # Binary labels: 1 if sweep < tc (or <= tc at the upper boundary)
        op = (lambda t: t <= tc) if np.isclose(tc, tc_max) else (lambda t: t < tc)
        y_tr, y_va, y_te = (op(t).astype(np.float32) for t in (t_tr, t_va, t_te))

        X_tr_t = torch.from_numpy(X_tr.reshape(-1, 1, L, L)).float()
        X_va_t = torch.from_numpy(X_va.reshape(-1, 1, L, L)).float()
        X_te_t = torch.from_numpy(X_te.reshape(-1, 1, L, L)).float()
        y_tr_t = torch.from_numpy(y_tr).float()
        y_va_t = torch.from_numpy(y_va).float()
        y_te_t = torch.from_numpy(y_te).float()

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                                  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t),
                                  batch_size=batch_size, shuffle=True)

        torch.manual_seed(seed)
        model = SimpleCNN(L=L, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=patience_lr)
        criterion = nn.BCELoss()

        ckpt = tc_dir / f"best_CNN_w{w:.2f}_a{alpha:.4f}_L{L}-{sID}.pt"
        history = train_one_run(
            model, optimizer, scheduler, criterion,
            train_loader, val_loader, num_epochs, patience_stop, ckpt,
        )
        np.save(tc_dir / "history.npy", history)

        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        with torch.no_grad():
            out = model(X_te_t)
        is_correct = ((out >= 0.5).float() == y_te_t).float()
        acc = is_correct.mean().item()
        test_accuracies.append(acc)
        print(f"  tc={tc:.4f}  test acc = {acc:.4f}")

    accs_path = output_dir / f"accs_w{w:.2f}_a{alpha:.4f}_L{L}_seed{seed}-{sID}.csv"
    np.savetxt(accs_path, test_accuracies)
    grid_path = output_dir / f"tc_grid_w{w:.2f}_a{alpha:.4f}_L{L}_seed{seed}-{sID}.csv"
    np.savetxt(grid_path, tc_values)
    print(f"Wrote {accs_path}")
    return accs_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv", type=Path, help="Combined snapshot CSV (last column is sweep param).")
    p.add_argument("output_dir", type=Path, help="Output directory.")
    p.add_argument("--tc-min", type=float, required=True)
    p.add_argument("--tc-max", type=float, required=True)
    p.add_argument("--num-tc", type=int, required=True)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--sID", type=int, default=0)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=150)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--initial-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience-lr", type=int, default=10)
    p.add_argument("--patience-stop", type=int, default=20)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run_lbc(
        input_csv=a.input_csv, output_dir=a.output_dir,
        w=a.w, alpha=a.alpha, L=a.L, sID=a.sID, seed=a.seed,
        tc_min=a.tc_min, tc_max=a.tc_max, num_tc=a.num_tc,
        batch_size=a.batch_size, num_epochs=a.num_epochs, dropout=a.dropout,
        initial_lr=a.initial_lr, weight_decay=a.weight_decay,
        patience_lr=a.patience_lr, patience_stop=a.patience_stop,
    )


if __name__ == "__main__":
    main()
