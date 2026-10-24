"""Known-answer test of the PhaseLearner k-means / sigmoid beta_c estimator on 2D Ising spins.

Data: ~/Documents/lbcfss/data/spins_L{L}.npy (L, L, 1000, 48 T), T uniform in [1.2, 3.4]
(midpoint 2.30 -- almost exactly T_c = 2.269, so the FULL grid cannot distinguish a real
estimate from a grid-midpoint artifact; the off-centre windows below can).

For each window the embedding (PCA(2), or PCA(50) -> t-SNE) is refit end-to-end on the
samples inside the window, clustered with the same KMeans settings as
ssh_ml_phase.{pca,tsne}_method, and the ordered fraction f(T) is sigmoid-fitted with
p0 = window midpoint. 'ordered' = the cluster with the lower mean T, so f falls with T.
Two inputs: 'raw' spins (Z2 symmetry unbroken: +m and -m are separate blobs) and
'gauge' spins (each configuration flipped so its magnetisation is >= 0).

A trustworthy estimator returns the same T on every window that contains T_c
(the L-dependent pseudo-critical point, ~2.35-2.45 for L = 12-20), and its cluster
share moves with the window. A midpoint artifact tracks 'mid' with share ~0.5.

usage: python lbc_validation/ising_known_answer_test.py {pca|tsne} [L=16] [samples_per_T=1000] [perplexity=30] [3win]
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
D = "/Users/gissa/Documents/lbcfss/data"
method = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 16
nper = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
perp = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
three = len(sys.argv) > 5 and sys.argv[5] == "3win"
TC = 2.269185
spins = np.load(f"{D}/spins_L{L}.npy"); temps = np.load(f"{D}/temps_L{L}.npy")
Lx, Ly, N, nT = spins.shape
rng = np.random.default_rng(0); keep = np.sort(rng.choice(N, nper, replace=False))
X = spins[:, :, keep, :].transpose(3, 2, 0, 1).reshape(nT * nper, Lx * Ly).astype(np.float32)
T = np.repeat(temps, nper)
mag = X.mean(1)
variants = {"raw": X, "gauge": X * np.where(mag >= 0, 1.0, -1.0)[:, None]}

def sig_down(t, tc, w): return 1.0 / (1.0 + np.exp((t - tc) / w))
def fit(u, f, p0):
    try:
        popt, pcov = curve_fit(sig_down, u, f, p0=[p0, 0.2], bounds=([u[2], 0.01], [u[-3], 5.0]), maxfev=10000)
        return popt[0], popt[1]
    except Exception: return np.nan, np.nan
def embed(V):
    V = V - V.mean(0)
    if method == "pca": return PCA(2).fit_transform(V)
    Z = PCA(min(50, V.shape[1])).fit_transform(V)
    return TSNE(2, perplexity=perp, init="pca", random_state=42, n_jobs=-1).fit_transform(Z)
def kmeans(E):
    if method == "pca": return KMeans(2, random_state=10, n_init=25, init="random").fit_predict(E)
    return KMeans(2, random_state=42, n_init=10).fit_predict(E)

lo, hi = temps.min(), temps.max()
wins = {"full": (lo, hi), "[1.2,2.9]": (lo, 2.9), "[1.5,3.4]": (1.5, hi), "low2/3": (lo, lo + 2 * (hi - lo) / 3),
        "high2/3": (lo + (hi - lo) / 3, hi), "[2.3,3.4] noTc": (2.3, hi), "[1.2,2.2] noTc": (lo, 2.2)}
if three: wins = {k: wins[k] for k in ("full", "low2/3", "high2/3")}
print(f"# Ising L={L}, {nper}/T, {nT} T in [{lo:.2f},{hi:.2f}], method={method}, perplexity={perp:g}, Tc={TC:.3f}", flush=True)
for vname, V in variants.items():
    for wname, (w0, w1) in wins.items():
        if vname == "raw" and method == "tsne" and wname != "full": continue
        t0 = time.time(); sel = (T >= w0 - 1e-9) & (T <= w1 + 1e-9)
        E = embed(V[sel]); Ts = T[sel]; lab = kmeans(E)
        ordered = 0 if Ts[lab == 0].mean() < Ts[lab == 1].mean() else 1
        u = np.unique(Ts); f = np.array([(lab[Ts == x] == ordered).mean() for x in u])
        tc, w = fit(u, f, (w0 + w1) / 2); share = (lab == ordered).mean()
        ntrans = int(np.sum((f > 0.1) & (f < 0.9)))
        extra = ""
        if wname == "full":
            hl = HDBSCAN(min_cluster_size=300, min_samples=20).fit_predict(StandardScaler().fit_transform(E))
            ids = [i for i in np.unique(hl) if i >= 0]
            extra = " HDBSCAN: %d cl, noise %.2f " % (len(ids), (hl < 0).mean()) + " ".join(
                f"[T {Ts[hl==i].min():.2f}-{Ts[hl==i].max():.2f} n={np.sum(hl==i)}]" for i in sorted(ids, key=lambda i: Ts[hl == i].mean())[:5])
        print(f"{vname:5s} {wname:15s} mid={(w0+w1)/2:.2f}  Tc_est={tc:5.2f} (w={w:4.2f}, n_trans={ntrans:2d}, share={share:.2f})  dev from Tc={tc-TC:+.2f}  [{time.time()-t0:.0f}s]{extra}", flush=True)
