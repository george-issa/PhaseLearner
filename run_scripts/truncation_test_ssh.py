"""Truncation test for the k-means / sigmoid beta_c estimators (km_sig) on stored SSH results.

Re-clusters the stored 2D points of results/method_compare/npz/{pca,umap,tsne}_results_<obs>_sid5.npz
inside asymmetric beta windows (full, lower 2/3, upper 2/3, lower half, upper half) and refits the
sigmoid with p0 = window midpoint. A real estimate stays put on every window that contains it;
a grid-midpoint artifact tracks the printed window midpoint ("bc|mid").

Caveat: the embedding itself is NOT refit (fixed PCA scores / UMAP / t-SNE coordinates, ~100 samples
per beta subsample). For an end-to-end version on a known answer see lbc_validation/ising_known_answer_test.py.
Background: obsidian/07_Failures_and_Negative_Results.md (2026-09-06 entries).

Usage (repo root):  conda run -n phaselearner python run_scripts/truncation_test_ssh.py
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
FSS={0.24:17.5,0.245:14.3,0.25:12.5,0.255:12.3}
def sig(b,bc,w): return 1/(1+np.exp(-(b-bc)/w))
def fit(u,f,p0):
    try: popt,_=curve_fit(sig,u,f,p0=[p0,2.0],bounds=([u[2],0.1],[u[-3],20]),maxfev=10000); return popt[0]
    except Exception: return np.nan
def load(m,obs,a):
    d=np.load(f"results/method_compare/npz/{m}_results_{obs}_sid5.npz",allow_pickle=True); k=f"a{a:.4f}"
    if m=="pca": X=np.column_stack([d[k+"_P1_sample"],d[k+"_P2_sample"]]); b=d[k+"_betas_sample"]
    else: X=np.column_stack([d[k+"_emb_x"],d[k+"_emb_y"]]); b=d[k+"_emb_betas"]
    return X,b,d[k+"_u_beta"],d[k+"_km_sig_popt"]
def cluster(m,X,b):
    if m=="pca": lab=KMeans(2,random_state=10,n_init=25,init='random').fit_predict(X)
    elif m=="umap": lab=KMeans(2,random_state=42,n_init=100).fit_predict(StandardScaler().fit_transform(X))
    else: lab=KMeans(2,random_state=42,n_init=10).fit_predict(X)
    kv=1 if b[lab==1].mean()>b[lab==0].mean() else 0
    u=np.unique(b); f=np.array([(lab[b==x]==kv).mean() for x in u]); return u,f,(lab==kv).mean()
for m,obs in [("pca","integrated_bond33_position"),("umap","integrated_bond33_position"),("tsne","integrated_bond33_position"),("umap","radial_disp"),("pca","bond33_momentum"),("tsne","greens22_momentum")]:
    for a in [0.24,0.2425,0.245,0.25,0.255]:
        X,b,u,popt=load(m,obs,a); lo,hi=u.min(),u.max(); rng=hi-lo
        wins={"full":(lo,hi),"low2/3":(lo,lo+2*rng/3),"high2/3":(lo+rng/3,hi),"lowhalf":(lo,lo+rng/2),"highhalf":(lo+rng/2,hi)}
        out=[]
        for name,(w0,w1) in wins.items():
            sel=(b>=w0-1e-9)&(b<=w1+1e-9); uu,ff,share=cluster(m,X[sel],b[sel]); bc=fit(uu,ff,(w0+w1)/2)
            out.append(f"{name}:{bc:5.1f}|{(w0+w1)/2:4.1f}")
        print(f"{m:4s} {obs:28s} a={a:.4f} FSS={FSS.get(a,float('nan')):4.1f} stored={popt[0]:5.1f}  "+"  ".join(out))
