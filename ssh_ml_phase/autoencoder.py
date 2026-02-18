"""
Autoencoder anomaly detection for phase transition detection.

Strategy
--------
Train the autoencoder on ALL snapshots with a **tight bottleneck** (latent_dim=2).
The small bottleneck forces the network to learn the dominant mode of variation
— the order parameter — so the latent space acts like a learned PCA.

beta_c is extracted from:
  1. **Derivative peak** — The steepest change in mean reconstruction error
     locates the transition, analogous to the derivative of <|P1|> in PCA.
     Gaussian smoothing (sigma=1) is applied before finding the peak to
     suppress noise.
  2. **Latent-space derivative** — d<|z_0|>/d(beta) peak (Gaussian-smoothed),
     where z_0 is the leading latent dimension (learned order parameter).
  3. **Variance peak** — Var(reconstruction_error) at each beta may peak at
     beta_c, but is noisier than the derivative methods.
  4. **Latent KMeans** — KMeans (k=2) on the 2D latent space (z0 vs z1).
     Identifies ordered (KVBS) and disordered (SM) phases by cluster
     membership, then fits a sigmoid to f_KVBS(beta) to extract beta_c.

Robustness:
  - Average over multiple random seeds.
  - Global standardization (zero mean, unit variance) for stable training.

Usage
-----
    python3.11 -m ssh_ml_phase.autoencoder                    # all alphas
    python3.11 -m ssh_ml_phase.autoencoder --alphas 0.240     # single alpha
    python3.11 -m ssh_ml_phase.autoencoder --n-seeds 5        # more seeds
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

from .config import ALPHAS, L, FSS_DICT, DEVICE
from .data import load_data, ensure_results_dir


def sigmoid_fit(beta, beta_c, w):
    """Standard logistic sigmoid: f = 1 / (1 + exp(-(beta - beta_c)/w))."""
    return 1.0 / (1.0 + np.exp(-(beta - beta_c) / w))


def decreasing_sigmoid(beta, beta_c, w, A, B):
    """Decreasing sigmoid: A / (1 + exp((beta - beta_c)/w)) + B.

    Used to fit mean reconstruction error, which is high in the disordered
    phase and low in the ordered phase, with an inflection at beta_c.
    """
    return A / (1.0 + np.exp((beta - beta_c) / w)) + B


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with a small latent bottleneck."""

    def __init__(self, L=12, latent_dim=2):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * L * L, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 32 * L * L)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, padding=1),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 32, self.L, self.L)
        return self.decoder_conv(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def _single_run(X_norm, betas, u_beta, latent_dim, nepochs, batch_size,
                lr, seed, verbose):
    """Train one AE and return per-sample errors and latent vectors."""
    X_t = torch.tensor(X_norm.reshape(-1, 1, L, L), dtype=torch.float32)

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
    model = ConvAutoencoder(L=L, latent_dim=latent_dim).to(DEVICE)
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
            errors = ((recon - xb) ** 2).mean(dim=(1, 2, 3))
            all_errors.append(errors.cpu().numpy())
            all_latent.append(z.cpu().numpy())

    return np.concatenate(all_errors), np.concatenate(all_latent)


def train_and_evaluate(alpha, latent_dim=2, nepochs=100, batch_size=64,
                       lr=1e-3, n_seeds=3, verbose=True):
    """
    Train autoencoder (multi-seed average) and extract beta_c.

    Trains on ALL data with a tight bottleneck (latent_dim=2). The bottleneck
    forces the AE to learn the dominant mode of variation (the order parameter).

    Parameters
    ----------
    n_seeds : int
        Number of random seeds to average over for robustness.
    """
    X, betas = load_data(alpha)
    u_beta = np.unique(betas)
    fss_bc = FSS_DICT.get(alpha, float(np.mean(u_beta)))

    # Global standardization (zero mean, unit variance per feature)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"  Training on all {len(u_beta)} beta values, {n_seeds} seeds...")

    # Run multiple seeds and collect per-sample errors
    all_seed_errors = []
    all_seed_latent = []
    for s in range(n_seeds):
        seed = 42 + s * 100
        errs, lats = _single_run(
            X_norm, betas, u_beta, latent_dim, nepochs, batch_size,
            lr, seed, verbose,
        )
        all_seed_errors.append(errs)
        all_seed_latent.append(lats)
        if verbose:
            me = np.array([np.mean(errs[betas == b]) for b in u_beta])
            d_me = np.abs(np.diff(me) / np.diff(u_beta))
            bm = 0.5 * (u_beta[:-1] + u_beta[1:])
            print(f"    seed {s}: deriv peak at beta={bm[np.argmax(d_me)]:.1f}")

    # Average reconstruction errors across seeds
    avg_errors = np.mean(all_seed_errors, axis=0)
    # Use latent from first seed for visualization and clustering
    latent = all_seed_latent[0]

    # ── Per-beta statistics ───────────────────────────────────────────────
    mean_error = np.array([np.mean(avg_errors[betas == b]) for b in u_beta])
    var_error  = np.array([np.var(avg_errors[betas == b])  for b in u_beta])
    std_error  = np.sqrt(var_error)

    # Latent stats
    var_z0      = np.array([np.var(latent[betas == b, 0])            for b in u_beta])
    mean_abs_z0 = np.array([np.mean(np.abs(latent[betas == b, 0]))   for b in u_beta])

    # ── Extract beta_c ────────────────────────────────────────────────────
    beta_mid = 0.5 * (u_beta[:-1] + u_beta[1:])

    # Method 1: Derivative of mean reconstruction error (Gaussian-smoothed)
    # Skip the first `deriv_skip` indices to avoid spurious peaks from
    # low-beta finite-difference noise (beta < ~3).
    deriv_skip   = 4
    d_err        = np.diff(mean_error) / np.diff(u_beta)
    d_err_smooth = gaussian_filter1d(np.abs(d_err), sigma=2)
    bc_deriv     = float(beta_mid[deriv_skip + np.argmax(d_err_smooth[deriv_skip:])])

    # Method 2: Derivative of mean |z_0| (learned order parameter, smoothed)
    d_z0        = np.diff(mean_abs_z0) / np.diff(u_beta)
    d_z0_smooth = gaussian_filter1d(np.abs(d_z0), sigma=2)
    bc_z0_deriv = float(beta_mid[deriv_skip + np.argmax(d_z0_smooth[deriv_skip:])])

    # Method 3: Variance peak of reconstruction error
    bc_var = u_beta[np.argmax(var_error)]

    # Method 5: Sigmoid fit to mean_error (most robust)
    # mean_error is a monotonically decreasing function (high in disordered
    # phase, low in ordered phase) — a decreasing sigmoid whose midpoint is
    # beta_c.  This uses ALL data points and is robust to low-beta noise.
    try:
        A0 = float(mean_error.max() - mean_error.min())
        B0 = float(mean_error.min())
        popt_err_sig, _ = curve_fit(
            decreasing_sigmoid, u_beta, mean_error,
            p0=[fss_bc, 3.0, A0, B0],
            bounds=([u_beta[3], 0.5, 0.0, -np.inf],
                    [u_beta[-3], 20.0, np.inf, np.inf]),
            maxfev=10000,
        )
        bc_err_sig = float(popt_err_sig[0])
    except Exception:
        popt_err_sig = None
        bc_err_sig = np.nan

    # Method 4: KMeans on 2D latent space (z0 vs z1)
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
            bc_lat_km = popt_lat[0]
        except Exception:
            popt_lat = None

        d_flat = np.diff(f_kvbs_lat) / np.diff(u_beta)
        bc_lat_km_deriv = beta_mid[np.argmax(d_flat)]

    # Best estimate: sigmoid fit to mean_error if converged, else derivative
    bc_best = bc_err_sig if not np.isnan(bc_err_sig) else bc_deriv

    return {
        "u_beta":       u_beta,
        "mean_error":   mean_error,
        "var_error":    var_error,
        "std_error":    std_error,
        "var_z0":       var_z0,
        "mean_abs_z0":  mean_abs_z0,
        "all_errors":   avg_errors,
        "latent":       latent,
        "betas":        betas,
        # beta_c estimates
        "beta_c_deriv":        bc_deriv,
        "beta_c_z0_deriv":     bc_z0_deriv,
        "beta_c_var":          bc_var,
        "beta_c_lat_km":       bc_lat_km,
        "beta_c_lat_km_deriv": bc_lat_km_deriv,
        "beta_c_err_sig":      bc_err_sig,
        "beta_c_best":         bc_best,
        # curves
        "deriv_curve":     (beta_mid, d_err),
        "z0_deriv_curve":  (beta_mid, d_z0),
        "d_err_smooth":    d_err_smooth,
        "d_z0_smooth":     d_z0_smooth,
        "err_sig_params":  popt_err_sig,
        # latent KMeans
        "lat_labels":  lat_labels,
        "f_kvbs_lat":  f_kvbs_lat,
        "sigmoid_lat": popt_lat,
    }


def run_all(alphas=None, latent_dim=2, n_seeds=3):
    """
    Run autoencoder analysis for all alphas. Saves to results/autoencoder_results.npz.

    beta_c array layout: [deriv, z0_deriv, var, lat_km_sig, lat_km_deriv, err_sig, best]
    """
    if alphas is None:
        alphas = ALPHAS

    results_dir = ensure_results_dir()
    all_results = {}

    for alpha in alphas:
        print(f"\n[Autoencoder] alpha={alpha:.3f} ...")
        res = train_and_evaluate(alpha, latent_dim=latent_dim, n_seeds=n_seeds)
        all_results[alpha] = res

        fss = FSS_DICT.get(alpha, np.nan)
        err_sig_str = f"{res['beta_c_err_sig']:.1f}" if not np.isnan(res['beta_c_err_sig']) else "nan"
        print(
            f"  beta_c: deriv={res['beta_c_deriv']:.1f}  "
            f"z0_deriv={res['beta_c_z0_deriv']:.1f}  "
            f"var={res['beta_c_var']:.1f}  "
            f"err_sig={err_sig_str}  "
            f"best={res['beta_c_best']:.1f}  FSS={fss:.1f}"
        )

    # Save
    save_dict = {}
    for alpha in alphas:
        r = all_results[alpha]
        key = f"a{alpha:.4f}"
        save_dict[f"{key}_u_beta"]      = r["u_beta"]
        save_dict[f"{key}_mean_error"]  = r["mean_error"]
        save_dict[f"{key}_var_error"]   = r["var_error"]
        save_dict[f"{key}_std_error"]   = r["std_error"]
        save_dict[f"{key}_var_z0"]      = r["var_z0"]
        save_dict[f"{key}_mean_abs_z0"] = r["mean_abs_z0"]
        # beta_c: [deriv, z0_deriv, var, lat_km_sig, lat_km_deriv, err_sig, best]
        save_dict[f"{key}_beta_c"] = np.array([
            r["beta_c_deriv"], r["beta_c_z0_deriv"], r["beta_c_var"],
            r["beta_c_lat_km"], r["beta_c_lat_km_deriv"],
            r["beta_c_err_sig"], r["beta_c_best"],
        ])
        save_dict[f"{key}_deriv_beta"]     = r["deriv_curve"][0]
        save_dict[f"{key}_deriv_vals"]     = r["deriv_curve"][1]
        save_dict[f"{key}_z0_deriv_vals"]  = r["z0_deriv_curve"][1]
        save_dict[f"{key}_d_err_smooth"]   = r["d_err_smooth"]
        save_dict[f"{key}_d_z0_smooth"]    = r["d_z0_smooth"]
        if r["err_sig_params"] is not None:
            save_dict[f"{key}_err_sig_params"] = np.array(r["err_sig_params"])
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

    np.savez(f"{results_dir}/autoencoder_results.npz", **save_dict)
    print(f"\nResults saved to {results_dir}/autoencoder_results.npz")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Autoencoder anomaly detection")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()
    run_all(alphas=args.alphas, latent_dim=args.latent_dim, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
