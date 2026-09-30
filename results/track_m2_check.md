# Track M v2, the observational check (500 histories, float64)

Each channel's lift fit checks two channels: the world's own, and the one his realistic observational specification reads off the history. Shares with their Clopper-Pearson 95 % intervals; a rejection is a p-value under 0.05.

| channel | truth rejected | covers 1 | observational rejected | covers its factor | floor holds | failed | unchecked |
|---|---|---|---|---|---|---|---|
| pla | 0.054 [0.036, 0.078] | 0.938 [0.913, 0.957] | 0.998 [0.989, 1.000] | 0.939 [0.914, 0.958] | 0.963 [0.942, 0.978] | 0 | 0 |
| meta | 0.052 [0.034, 0.075] | 0.958 [0.937, 0.974] | 0.992 [0.980, 0.998] | 0.961 [0.939, 0.976] | 0.979 [0.962, 0.990] | 0 | 0 |
| tv | 0.038 [0.023, 0.059] | 0.954 [0.932, 0.971] | 0.976 [0.958, 0.988] | 0.986 [0.923, 1.000] | 1.000 [0.949, 1.000] | 0 | 0 |

The observational channel: the median factor the tests read of it, the median factor without noise, and its least `Gamma` at an effect-scale gap of 1, median and the share infinite (no level reconciles it with the tests); over the channels that predict a gap, and the count that predict none.

| channel | factor | noise-free | least Gamma | infinite | no factor |
|---|---|---|---|---|---|
| pla | 0.505 | 0.506 | 2.75 | 0.02 | 11 |
| meta | 0.181 | 0.180 | 7.82 | 0.01 | 17 |
| tv | 1.966 | 2.008 | 2.45 | 0.33 | 430 |

**Gate** (pla: the truth rejected in at most 0.07, both factors' intervals covering in at least 0.93, and the observational channel rejected in at least 0.90): met.
