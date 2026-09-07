# P2 tables -- fold design for cross-fitting on networks and panels

120 paired draws per arm, 10000 paired bootstrap resamples, 95% percentile intervals. Ratios are mean squared error against the random-unit split on the same draws.

## Table 1 -- fold schemes on the cycle, g = 2

| split | direct | spillover |
|---|---|---|
| random rows | 0.861 [0.692, 1.076] | 0.976 [0.847, 1.120] |
| random units | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| contiguous folds | 1.370 [1.101, 1.732] | 1.357 [1.145, 1.614] |
| neighbour exclusion | 1.666 [1.276, 2.206] | 1.612 [1.307, 1.972] |
| designed folds | 0.906 [0.776, 1.065] | 0.995 [0.888, 1.105] |

## Table 2 -- the design effect is O(1/g)

| split | g = 2 | g = 4 | g = 8 | g = 20 |
|---|---|---|---|---|
| random rows | 0.861 [0.692, 1.076] | 0.931 [0.797, 1.097] | 0.979 [0.840, 1.135] | 0.993 [0.914, 1.075] |
| contiguous folds | 1.370 [1.101, 1.732] | 1.139 [0.967, 1.355] | 1.083 [0.911, 1.304] | 1.047 [0.970, 1.133] |
| neighbour exclusion | 1.666 [1.276, 2.206] | 1.240 [1.030, 1.516] | 1.241 [1.027, 1.524] | 1.078 [0.976, 1.198] |
| designed folds | 0.906 [0.776, 1.065] | 0.911 [0.781, 1.060] | 0.895 [0.775, 1.043] | 0.977 [0.905, 1.059] |

Direct coefficient; the spillover column is in the JSON. The ordering is invariant in `g` (Result 60) while the size decays.

## Table 3 -- the law forecasts WHICH topology has a design effect

| topology | predicted mass ratio | realised MSE, direct | realised MSE, spillover |
|---|---|---|---|
| cycle | 0.720 | 0.661 [0.524, 0.835] | 0.733 [0.624, 0.853] |
| torus | 0.974 | 0.914 [0.764, 1.093] | 1.007 [0.907, 1.126] |
| cubic | 0.966 | 1.151 [0.956, 1.385] | 0.964 [0.873, 1.064] |

Designed against contiguous, so a ratio below 1 is a win for the design.

## Table 4 -- the q = 3 matrix moment: RELATIVE max-entry error against the exact anchor

| nodes | points | n = 5 (boundary) | n = 7 (margin 2) | residual / true, n = 7 |
|---|---|---|---|---|
| 4 | 4,096 | 2.923e-01 | 2.866e-01 | -- |
| 5 | 15,625 | 4.683e-02 | 1.702e-01 | 2.06 |
| 6 | 46,656 | 8.662e-02 | 5.594e-02 | 3.64 |

Relative throughout: the exact answer is `I` at `n = 5` and `I/3` at `n = 7`, so an absolute column would flatter the second by 3x (Result 63 (j)).
