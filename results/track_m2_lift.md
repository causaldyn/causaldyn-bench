# Track M v2, lift tests (500 histories a row, float64)

Coverage of each parameter's 95 % profile-likelihood interval (Clopper-Pearson 95 % in brackets), the share of intervals closed on both sides, the median width of those, and the fits that raised rather than converge, which cover nothing.

| setting | retention | closed | width | scale | closed | coefficient | closed | failed |
|---|---|---|---|---|---|---|---|---|
| pla, 4 tests, noise 1% | 0.954 [0.932, 0.971] | 0.81 | 0.242 | 0.976 | 0.17 | 0.978 | 0.83 | 0 |
| pla, 2 tests, noise 1% | 0.948 [0.925, 0.966] | 0.54 | 0.306 | 0.970 | 0.10 | 0.974 | 0.75 | 0 |
| pla, 4 tests, noise 3% | 0.958 [0.937, 0.974] | 0.15 | 0.554 | 0.986 | 0.01 | 0.970 | 0.56 | 0 |
| pla, 4 tests, noise 10% | 0.878 [0.846, 0.905] | 0.02 | 0.244 | 0.962 | 0.03 | 0.878 | 0.38 | 4 |
| meta, 4 tests, noise 1% | 0.918 [0.890, 0.941] | 0.64 | 0.465 | 0.970 | 0.04 | 0.932 | 0.47 | 0 |
| tv, 4 tests, noise 1% | 0.958 [0.937, 0.974] | 0.38 | 0.479 | 0.970 | 0.04 | 0.938 | 0.56 | 0 |

Median estimate against the truth, and the median of the adstock range the readouts covered (the curve is read over that range only).

| setting | retention | scale | coefficient | tested adstock |
|---|---|---|---|---|
| pla, 4 tests, noise 1% | 0.1991 (0.2) | 257.1 (250) | 1144 (1100) | 0.1 to 192.3 |
| pla, 2 tests, noise 1% | 0.1981 (0.2) | 247.9 (250) | 1102 (1100) | 0.1 to 182.1 |
| pla, 4 tests, noise 3% | 0.1738 (0.2) | 273.9 (250) | 1286 (1100) | 0.0 to 192.0 |
| pla, 4 tests, noise 10% | 0.09712 (0.2) | 148.9 (250) | 1665 (1100) | 0.0 to 191.7 |
| meta, 4 tests, noise 1% | 0.4101 (0.4) | 191.3 (200) | 592.6 (600) | 0.9 to 114.4 |
| tv, 4 tests, noise 1% | 0.7318 (0.7) | 306.5 (333.3) | 507.2 (560) | 1.4 to 142.6 |

**Gate** (pla, 4 tests, noise 1%, retention within 0.02 of 0.95): met.
