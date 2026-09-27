# P4 tables -- residuals go blind near degeneracy

The anti-monotone LQ mean-field game (coupling 3, obstruction `T* = 0.8036`), one fixed box of half-width 10, one Deep Galerkin solve per horizon. Ranks are Spearman's, each score against `|S_hat(0) - S(0)|` over the horizons.

## Table 1 -- one solve per horizon at the defaults (seed 0, width 32, Adam 2500)

| T | den(T) | S(0) exact | S_hat(0) | error | control error | raw residual | residual / abs(den) | dual-weighted |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.7072 | -2.8010 | -2.5377 | 0.2633 | 9.1% | 3.591e-02 | 5.078e-02 | 0.2644 |
| 0.40 | 0.5824 | -3.3558 | -2.4203 | 0.9355 | 29.0% | 8.632e-02 | 1.482e-01 | 0.9379 |
| 0.50 | 0.4473 | -4.2330 | -2.2550 | 1.9780 | 51.4% | 1.048e-01 | 2.342e-01 | 1.9799 |
| 0.60 | 0.3045 | -5.9107 | -2.2551 | 3.6555 | 71.5% | 9.542e-02 | 3.134e-01 | 3.6554 |
| 0.68 | 0.1863 | -9.1375 | -2.2331 | 6.9044 | 90.2% | 6.038e-02 | 3.242e-01 | 6.8858 |
| 0.72 | 0.1263 | -13.0333 | -2.2052 | 10.8281 | 100.3% | 1.817e-02 | 1.439e-01 | 10.7666 |
| 0.76 | 0.0659 | -24.0370 | -2.1735 | 21.8635 | 110.9% | 6.973e-03 | 1.057e-01 | 21.4922 |
| 0.79 | 0.0206 | -74.7280 | -2.1464 | 72.5816 | 119.1% | 1.375e-02 | 6.689e-01 | 68.2083 |

Rank correlation with the error: raw residual **-0.667**, conditioned **+0.405**, dual-weighted **+1.000**; worst relative discrepancy of the dual-weighted estimate `0.0603`.

## Table 2 -- the same ranks over seeds, widths and optimisers (min / median / max)

| optimiser | width | seeds | raw residual | residual / abs(den) | dual-weighted | worst discrepancy | largest abs(S_hat(0)) |
|---|---|---|---|---|---|---|---|
| adam | 32 | 5 | -0.667 / -0.667 / -0.667 | +0.381 / +0.405 / +0.500 | +1.000 / +1.000 / +1.000 | 0.0708 | 2.62 |
| adam | 64 | 5 | -0.476 / -0.452 / -0.286 | +0.500 / +0.619 / +0.619 | +1.000 / +1.000 / +1.000 | 0.0869 | 2.88 |
| adam | 128 | 5 | -0.452 / -0.286 / -0.238 | +0.619 / +0.619 / +0.690 | +1.000 / +1.000 / +1.000 | 0.1201 | 2.99 |
| lbfgs | 32 | 5 | -0.024 / +0.619 / +0.714 | +0.548 / +0.905 / +0.929 | +1.000 / +1.000 / +1.000 | 0.2938 | 5.07 |
| lbfgs | 64 | 5 | -0.024 / +0.714 / +0.810 | +0.619 / +0.905 / +0.952 | +0.976 / +1.000 / +1.000 | 0.6938 | 5.67 |
| lbfgs | 128 | 5 | +0.262 / +0.595 / +0.929 | +0.667 / +0.833 / +0.976 | +0.976 / +1.000 / +1.000 | 0.7977 | 7.03 |

Pooled over all 30 configurations: raw residual -0.667 / -0.131 / +0.929, conditioned +0.381 / +0.619 / +0.976, dual-weighted +0.976 / +1.000 / +1.000. The raw residual correlates negatively with the error in 17 of 30 and fails to correlate positively in 17; the dual-weighted estimate ranks it perfectly in 28, and is never further than `0.7977` from it in relative terms.
