# LBC validation — 2D Ising

A regression test for the learning-by-confusion pipeline in `../lbc_train.py`.
It runs LBC against a model with a known critical point — the 2D square Ising
model, exact `Tc = 2/ln(1+sqrt2) = 2.269 J` — so any change to the CNN, the
augmentation, or the sweep logic can be checked against the right answer.

## Run

```bash
./run_ising_check.sh                 # writes to ../results/lbc_ising_check/
PYTHON=/path/to/python ./run_ising_check.sh   # override interpreter
```

Three steps, ~10 min on a laptop CPU:

1. `make_ising_csv.py` — checkerboard-Metropolis snapshots, 4x4, T in [1.0, 3.6].
2. `lbc_train.py` — LBC sweep, one CNN retrained per guessed `tc` (27 total).
3. `plot_lbc.py` — the W-curve and the estimated `Tc`.

## Expected result

A textbook **W-shape**: ~100% accuracy at both arms (trivial splits), a dip on
either side, and a **central peak on top of Tc**. On 4x4 the peak lands at
`tc = 2.20`, within one grid step (0.1) of the exact 2.269.

Reference figure: `../results/lbc_ising_check/lbc_ising_L4_Wcurve.png`.

## Reading the curve

The LBC estimator is the accuracy maximum **between the two W-minima**, not the
global interior maximum — near the arms the split is almost trivial, so a plain
`argmax` reports a spurious value close to an endpoint. `plot_lbc.py` finds each
dip first, then the peak strictly between them.

Bulky regenerable outputs (raw snapshots, augmented data, per-`tc` checkpoints)
land in `../results/lbc_ising_check/work/`, which is gitignored.
