# P2 tables -- fold design for cross-fitting on networks and panels

120 paired draws per arm, 10000 paired bootstrap resamples, 95% percentile intervals. Ratios are mean squared error against the random-unit split on the same draws.

## Table 1 -- fold schemes on the cycle, g = 2

| split | direct | spillover |
|---|---|---|
| random rows | 0.877 [0.702, 1.088] | 0.918 [0.812, 1.035] |
| random units | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| contiguous folds | 1.221 [0.933, 1.591] | 1.336 [1.136, 1.556] |
| neighbour exclusion | 1.705 [1.285, 2.245] | 1.651 [1.336, 2.041] |
| designed folds | 0.843 [0.704, 1.012] | 1.008 [0.908, 1.119] |

## Table 2 -- the design effect is O(1/g)

| split | g = 2 |
|---|---|
| random rows | 0.877 [0.702, 1.088] |
| contiguous folds | 1.221 [0.933, 1.591] |
| neighbour exclusion | 1.705 [1.285, 2.245] |
| designed folds | 0.843 [0.704, 1.012] |

Direct coefficient; the spillover column is in the JSON. The ordering is invariant in `g` (Result 60) while the size decays.

## Table 3 -- the law forecasts WHICH topology has a design effect

| topology | predicted mass ratio | realised MSE, direct | realised MSE, spillover |
|---|---|---|---|
| cycle | 0.720 | 0.690 [0.537, 0.888] | 0.754 [0.631, 0.908] |
| torus | 0.974 | 1.092 [0.875, 1.345] | 0.989 [0.925, 1.059] |
| cubic | 0.966 | 1.107 [0.886, 1.367] | 0.927 [0.837, 1.021] |

Designed against contiguous, so a ratio below 1 is a win for the design.

## Table 4 -- the q = 3 matrix moment: RELATIVE max-entry error against the exact anchor

| nodes | points | n = 5 (boundary) | rate | residual / true | n = 7 (margin 2) | rate | residual / true |
|---|---|---|---|---|---|---|---|
| 4 | 4,096 | 2.923e-01 | -- | -- | 2.866e-01 | -- | -- |

Relative throughout: the exact answer is `I` at `n = 5` and `I/3` at `n = 7`, so an absolute column would flatter the second by 3x (Result 63 (j)). The rate is the per-node decay `e(nodes - 1) / e(nodes)`. By the reverse triangle inequality residual / true is at least `|rate - 1|`, with equality when the error keeps its sign and its largest entry: a rate of 2 always makes the residual majorise the error, and below 2 only a row where that premise fails can (Result 63 (e)).
