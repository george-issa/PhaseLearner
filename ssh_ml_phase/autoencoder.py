"""
Autoencoder anomaly detection for phase transition detection.

Architecture
------------
A fully-connected (MLP) autoencoder with a tight 2D bottleneck is used instead
of a CNN.  The bond-bond data lives in *momentum space* — there is no meaningful
spatial locality in the Brillouin zone, so convolutional filters would be
physically unmotivated.  An MLP treats all k-points on equal footing.

Strategy
--------
Train on ALL snapshots with a latent_dim=2 bottleneck.  The bottleneck forces
the network to learn the dominant mode of variation — the order parameter —
so the latent space acts like a learned, non-linear PCA.

beta_c extraction
-----------------
1. **Var(z0) susceptibility** (primary) — Var(z0)(beta) peaks at beta_c, exactly
   like the PCA susceptibility Var(P1)(beta).  Gaussian smoothing + smooth argmax
   gives the most robust estimate.
2. **Latent KMeans** — KMeans (k=2) on the 2D latent space.  Sigmoid fit to
   f_KVBS(beta) and its derivative peak give two additional estimates.
3. **Reconstruction error** (secondary) — mean reconstruction error is
   monotonically decreasing; its smoothed derivative peak is a fallback.

Usage
-----
    python3.11 -m ssh_ml_phase.autoencoder                              # all alphas
    python3.11 -m ssh_ml_phase.autoencoder --alphas 0.240              # single alpha
    python3.11 -m ssh_ml_phase.autoencoder --n-seeds 5                 # more seeds
    python3.11 -m ssh_ml_phase.autoencoder --beta-min 8.0              # skip low-beta region
    python3.11 -m ssh_ml_phase.autoencoder --file-template "GF_L12_a{alpha:.4f}.csv"  # new observable
    python3.11 -m ssh_ml_phase.autoencoder --tag bmin8                 # explicit output label
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import ALPHAS, FSS_DICT, DEVICE
from .data import load_beta_sweep, ensure_results_dir, make_tag, result_filename


def sigmoid_fit(beta, beta_c, w):
    """Standard logistic sigmoid: f = 1 / (1 + exp(-(beta - beta_c)/w))."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


def decreasing_sigmoid(beta, beta_c, w, A, B):
    """Decreasing sigmoid: A / (1 + exp((beta - beta_c)/w)) + B.

    Used to fit mean reconstruction error, which is high in the disordered
    phase and low in the ordered phase, with an inflection at beta_c.
    """
    return A / (1.0 + np.exp((beta - beta_c) / w)) + B


class MLPAutoencoder(nn.Module):
    """Fully-connected autoencoder with a small latent bottleneck.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input (number of k-space features).
    latent_dim : int
        Bottleneck dimension (default 2).
    """

    def __init__(self, n_features, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n_features),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def _single_run(X_norm, betas, u_beta, n_features, latent_dim, nepochs,
                batch_size, lr, seed, verbose):
    """Train one AE and return per-sample errors and latent vectors."""
    X_t = torch.tensor(X_norm, dtype=torch.float32)  # shape (N, n_features)

    n_val = max(int(0.1 * len(X_t)), 1)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X_t), generator=gen)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(X_t[train_idx]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_t[val_idx]), batch_size=batch_size
    )

    torch.manual_seed(seed)
    model = MLPAutoencoder(n_features=n_features, latent_dim=latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(nepochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                recon, _ = model(xb)
                val_loss += criterion(recon, xb).item() * len(xb)
        val_loss /= n_val
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 20:
            break

    model.load_state_dict(best_state)
    model.eval()
    all_errors = []
    all_latent = []
    full_loader = DataLoader(TensorDataset(X_t), batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in full_loader:
            xb = xb.to(DEVICE)
            recon, z = model(xb)
            errors = ((recon - xb) ** 2).mean(dim=1)  # per-sample MSE
            all_errors.append(errors.cpu().numpy())
            all_latent.append(z.cpu().numpy())

    return np.concatenate(all_errors), np.concatenate(all_latent)


def train_and_evaluate(alpha, latent_dim=2, nepochs=100, batch_size=64,
                       lr=1e-3, n_seeds=3, verbose=True,
                       beta_min=None, beta_max=None, stride=None,
                       file_template=None):
    """
    Train MLP autoencoder (multi-seed average) and extract beta_c.

    Parameters
    ----------
    beta_min : float or None
        Discard samples with beta < beta_min before training.
    file_template : str or None
        Override the default filename template.
    n_seeds : int
        Number of random seeds to average over for robustness.
    """
    X, betas = load_beta_sweep(alpha, beta_min=beta_min, beta_max=beta_max,
                               stride=stride, file_template=file_template)
    u_beta = np.unique(betas)
    n_features = X.shape[1]
    fss_bc = FSS_DICT.get(alpha, float(np.mean(u_beta)))

    # Global standardization (zero mean, unit variance per feature)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"  n_features={n_features}, n_beta={len(u_beta)}, "
              f"N={len(X)}, {n_seeds} seeds...")

    # ── Multi-seed training ───────────────────────────────────────────────────
    all_seed_errors = []
    all_seed_latent = []
    for s in range(n_seeds):
        seed = 42 + s * 100
        errs, lats = _single_run(
            X_norm, betas, u_beta, n_features, latent_dim, nepochs,
            batch_size, lr, seed, verbose,
        )
        all_seed_errors.append(errs)
        all_seed_latent.append(lats)
        if verbose:
            var_z0 = np.array([np.var(lats[betas == b, 0]) for b in u_beta])
            var_z0_sm = gaussian_filter1d(var_z0, sigma=2)
            bc_var = float(u_beta[np.argmax(var_z0_sm)])
            print(f"    seed {s}: Var(z0) peak at beta={bc_var:.1f}")

    # Average reconstruction errors; use first-seed latent for clustering
    avg_errors = np.mean(all_seed_errors, axis=0)
    latent = all_seed_latent[0]

    # ── Per-beta statistics ───────────────────────────────────────────────────
    mean_error = np.array([np.mean(avg_errors[betas == b]) for b in u_beta])
    var_error  = np.array([np.var(avg_errors[betas == b])  for b in u_beta])
    std_error  = np.sqrt(var_error)

    var_z0      = np.array([np.var(latent[betas == b, 0])          for b in u_beta])
    mean_abs_z0 = np.array([np.mean(np.abs(latent[betas == b, 0])) for b in u_beta])

    # ── Method 1: Var(z0) susceptibility — primary estimator ─────────────────
    # Exactly analogous to the PCA susceptibility Var(P1).  The variance of the
    # leading latent coordinate peaks at beta_c where the bottleneck is most
    # uncertain about how to compress the configuration.
    var_z0_sm       = gaussian_filter1d(var_z0, sigma=2)
    bc_var_z0_sm    = float(u_beta[np.argmax(var_z0_sm)])

    beta_mid = 0.5 * (u_beta[:-1] + u_beta[1:])
    d_var_z0_sm = np.diff(var_z0_sm) / np.diff(u_beta)

    # ── Method 2: Reconstruction error derivative (fallback) ─────────────────
    deriv_skip   = 4
    d_err        = np.diff(mean_error) / np.diff(u_beta)
    d_err_smooth = gaussian_filter1d(np.abs(d_err), sigma=2)
    bc_deriv     = float(beta_mid[deriv_skip + np.argmax(d_err_smooth[deriv_skip:])])

    # ── Method 3: Latent KMeans ───────────────────────────────────────────────
    bc_lat_km       = np.nan
    bc_lat_km_deriv = np.nan
    f_kvbs_lat      = None
    popt_lat        = None
    lat_labels      = None

    if latent.shape[1] >= 2:
        km_lat = KMeans(n_clusters=2, random_state=42, n_init=10)
        lat_labels = km_lat.fit_predict(latent[:, :2])

        mean_beta_l0 = np.mean(betas[lat_labels == 0])
        mean_beta_l1 = np.mean(betas[lat_labels == 1])
        kvbs_label_lat = 1 if mean_beta_l1 > mean_beta_l0 else 0

        f_kvbs_lat = np.array(
            [np.mean(lat_labels[betas == b] == kvbs_label_lat) for b in u_beta]
        )

        try:
            popt_lat, _ = curve_fit(
                sigmoid_fit, u_beta, f_kvbs_lat,
                p0=[fss_bc, 2.0],
                bounds=([u_beta[2], 0.1], [u_beta[-3], 20.0]),
                maxfev=10000,
            )
            bc_lat_km = float(popt_lat[0])
        except Exception:
            popt_lat = None

        d_flat = np.diff(f_kvbs_lat) / np.diff(u_beta)
        bc_lat_km_deriv = float(beta_mid[np.argmax(d_flat)])

    # ── Best estimate: Var(z0) smooth argmax ─────────────────────────────────
    bc_best = bc_var_z0_sm

    return {
        "u_beta":       u_beta,
        "mean_error":   mean_error,
        "var_error":    var_error,
        "std_error":    std_error,
        "var_z0":       var_z0,
        "var_z0_sm":    var_z0_sm,
        "mean_abs_z0":  mean_abs_z0,
        "all_errors":   avg_errors,
        "latent":       latent,
        "betas":        betas,
        # beta_c estimates
        "beta_c_var_z0_sm":    bc_var_z0_sm,
        "beta_c_deriv":        bc_deriv,
        "beta_c_lat_km":       bc_lat_km,
        "beta_c_lat_km_deriv": bc_lat_km_deriv,
        "beta_c_best":         bc_best,
        # curves
        "var_z0_deriv_curve":  (beta_mid, d_var_z0_sm),
        "deriv_curve":         (beta_mid, d_err),
        "d_err_smooth":        d_err_smooth,
        # latent KMeans
        "lat_labels":  lat_labels,
        "f_kvbs_lat":  f_kvbs_lat,
        "sigmoid_lat": popt_lat,
    }


def run_all(alphas=None, latent_dim=2, n_seeds=3, beta_min=None, beta_max=None,
            stride=None, file_template=None, tag=None):
    """
    Run autoencoder analysis for all alphas. Saves to results/autoencoder_results.npz.

    beta_c array layout: [var_z0_sm, deriv, lat_km_sig, lat_km_deriv, best]
    """
    if alphas is None:
        alphas = ALPHAS

    results_dir = ensure_results_dir()
    all_results = {}

    for alpha in alphas:
        print(f"\n[Autoencoder] alpha={alpha:.3f} ...")
        res = train_and_evaluate(
            alpha, latent_dim=latent_dim, n_seeds=n_seeds,
            beta_min=beta_min, beta_max=beta_max, stride=stride,
            file_template=file_template,
        )
        all_results[alpha] = res

        fss = FSS_DICT.get(alpha, np.nan)
        print(
            f"  beta_c: var_z0={res['beta_c_var_z0_sm']:.1f}  "
            f"deriv={res['beta_c_deriv']:.1f}  "
            f"lat_km={res['beta_c_lat_km']:.1f}  "
            f"best={res['beta_c_best']:.1f}  FSS={fss:.1f}"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]      = r["u_beta"]
        save_dict[f"{key}_mean_error"]  = r["mean_error"]
        save_dict[f"{key}_var_error"]   = r["var_error"]
        save_dict[f"{key}_std_error"]   = r["std_error"]
        save_dict[f"{key}_var_z0"]      = r["var_z0"]
        save_dict[f"{key}_var_z0_sm"]   = r["var_z0_sm"]
        save_dict[f"{key}_mean_abs_z0"] = r["mean_abs_z0"]
        # beta_c: [var_z0_sm, deriv, lat_km_sig, lat_km_deriv, best]
        save_dict[f"{key}_beta_c"] = np.array([
            r["beta_c_var_z0_sm"], r["beta_c_deriv"],
            r["beta_c_lat_km"], r["beta_c_lat_km_deriv"],
            r["beta_c_best"],
        ])
        save_dict[f"{key}_var_z0_deriv_beta"] = r["var_z0_deriv_curve"][0]
        save_dict[f"{key}_var_z0_deriv_vals"] = r["var_z0_deriv_curve"][1]
        save_dict[f"{key}_deriv_beta"]        = r["deriv_curve"][0]
        save_dict[f"{key}_deriv_vals"]        = r["deriv_curve"][1]
        save_dict[f"{key}_d_err_smooth"]      = r["d_err_smooth"]
        # Latent KMeans
        if r["f_kvbs_lat"] is not None:
            save_dict[f"{key}_f_kvbs_lat"] = r["f_kvbs_lat"]
        if r["sigmoid_lat"] is not None:
            save_dict[f"{key}_sigmoid_lat"] = np.array(r["sigmoid_lat"])
        # Latent space samples
        idx = np.random.default_rng(42).choice(
            len(r["latent"]), min(5000, len(r["latent"])), replace=False
        )
        save_dict[f"{key}_z0_sample"]    = r["latent"][idx, 0]
        if r["latent"].shape[1] > 1:
            save_dict[f"{key}_z1_sample"] = r["latent"][idx, 1]
        save_dict[f"{key}_betas_sample"] = r["betas"][idx]
        if r["lat_labels"] is not None:
            save_dict[f"{key}_lat_km_sample"] = r["lat_labels"][idx]
        for d in range(min(4, r["latent"].shape[1])):
            mean_z = np.array(
                [np.mean(np.abs(r["latent"][r["betas"] == b, d])) for b in r["u_beta"]]
            )
            save_dict[f"{key}_mean_z{d}"] = mean_z

    fname = result_filename("autoencoder", make_tag(beta_min, file_template, tag,
                                                   beta_max=beta_max, stride=stride))
    np.savez(f"{results_dir}/{fname}", **save_dict)
    print(f"\nResults saved to {results_dir}/{fname}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Autoencoder anomaly detection")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--beta-min", type=float, default=None,
                        help="Discard samples with beta < this value")
    parser.add_argument("--beta-max", type=float, default=None,
                        help="Discard samples with beta > this value")
    parser.add_argument("--stride", type=int, default=None,
                        help="Use every N-th row (subsample dense datasets)")
    parser.add_argument("--file-template", type=str, default=None,
                        help="Filename template with {alpha:.4f} placeholder")
    parser.add_argument("--tag", type=str, default=None,
                        help="Label appended to output filename (auto-generated if omitted)")
    args = parser.parse_args()
    run_all(alphas=args.alphas, latent_dim=args.latent_dim, n_seeds=args.n_seeds,
            beta_min=args.beta_min, beta_max=args.beta_max, stride=args.stride,
            file_template=args.file_template, tag=args.tag)


if __name__ == "__main__":
    main()
