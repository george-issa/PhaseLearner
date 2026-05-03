"""
Greitemann-Iqbal (Assaad) method for phase transition detection.

For each alpha, sweep guessed beta_c values. For each guess, train a CNN
to classify configurations from beta values on either side of the guess
(excluding `nskip` points around it). The peak test accuracy identifies
beta_c — the network distinguishes phases most easily when the guess is
correct and both training sides contain pure-phase data.

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
from .data import load_beta_sweep, ensure_npz_dir, result_filename


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


def run_single(alpha, nskip, nepochs=60, batch_size=32, patience=15, verbose=False,
               beta_min=None, beta_max=None, stride=None, file_template=None):
    """
    Run the Assaad method for one (alpha, nskip) pair.

    Returns
    -------
    trunc_betas : ndarray — guessed beta_c values
    test_accs   : ndarray — test accuracy at each guess
    """
    L = _cfg.L
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    u_beta = np.unique(betas)
    NL = len(u_beta)
    half = nskip // 2
    NG = NL - nskip
    trunc_beta = u_beta[half:-half]

    test_accuracies = []

    for iL in range(NG):
        beta_left = u_beta[iL : iL + half]
        beta_right = u_beta[iL + half + 1 : iL + nskip + 1]

        mask_left = np.isin(betas, beta_left)
        mask_right = np.isin(betas, beta_right)

        X0 = X[mask_left]
        X1 = X[mask_right]

        # Balance classes
        n_min = min(len(X0), len(X1))
        X0, X1 = X0[:n_min], X1[:n_min]

        X_all = np.vstack([X0, X1])
        y_all = np.concatenate([np.zeros(n_min), np.ones(n_min)])

        perm = np.random.permutation(len(X_all))
        X_all, y_all = X_all[perm], y_all[perm]

        Ntot = len(X_all)
        n_train = int(0.8 * Ntot)
        n_val = int(0.1 * Ntot)

        X_tr = torch.tensor(X_all[:n_train].reshape(-1, 1, L, L), dtype=torch.float32)
        y_tr = torch.tensor(y_all[:n_train], dtype=torch.long)
        X_va = torch.tensor(X_all[n_train:n_train + n_val].reshape(-1, 1, L, L), dtype=torch.float32)
        y_va = torch.tensor(y_all[n_train:n_train + n_val], dtype=torch.long)
        X_te = torch.tensor(X_all[n_train + n_val:].reshape(-1, 1, L, L), dtype=torch.float32)
        y_te = torch.tensor(y_all[n_train + n_val:], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size)

        torch.manual_seed(42)
        model = CNN_GI(L).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(nepochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            val_loss /= len(X_va)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = model(X_te.to(DEVICE)).argmax(dim=1).cpu()
            acc = (preds == y_te).float().mean().item()

        test_accuracies.append(acc)
        if verbose:
            print(f"  iL={iL}/{NG-1}  guess={trunc_beta[iL]:.1f}  acc={acc:.4f}")

    return trunc_beta, np.array(test_accuracies)


def run_all(alphas=None, nskip_values=None, tag=None,
            beta_min=None, beta_max=None, stride=None, file_template=None):
    """
    Run Assaad method for all alphas and nskip values.
    Saves to results/npz/assaad_results.npz.
    """
    if alphas is None:
        alphas = _cfg.ALPHAS
    if nskip_values is None:
        nskip_values = [4, 6, 8]

    results_dir = ensure_npz_dir()
    all_results = {}

    for alpha in alphas:
        all_results[alpha] = {}
        for nskip in nskip_values:
            print(f"[Assaad] alpha={alpha:.3f}, nskip={nskip} ...")
            trunc_betas, accs = run_single(alpha, nskip, verbose=False,
                                           beta_min=beta_min, beta_max=beta_max,
                                           stride=stride, file_template=file_template)
            all_results[alpha][nskip] = {"betas": trunc_betas, "accs": accs}

            bc_est = trunc_betas[np.argmax(accs)]
            fss = _cfg.FSS_DICT.get(alpha, np.nan)
            print(f"  beta_c={bc_est:.1f}  peak_acc={np.max(accs):.4f}  FSS={fss:.1f}")

    # Save
    save_dict = {}
    for alpha in alphas:
        for nskip in nskip_values:
            r = all_results[alpha][nskip]
            key = f"a{alpha:.4f}_ns{nskip}"
            save_dict[f"{key}_betas"] = r["betas"]
            save_dict[f"{key}_accs"] = r["accs"]
            save_dict[f"{key}_beta_c"] = np.array(
                [r["betas"][np.argmax(r["accs"])]]
            )

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
            beta_min=args.beta_min, beta_max=args.beta_max, stride=args.stride,
            file_template=file_template)


if __name__ == "__main__":
    main()
