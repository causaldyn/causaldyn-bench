# Track M v2, the observational check, pilot (100 histories, float64)

Each channel's lift fit checks two channels: the world's own, and the one his realistic observational specification reads off the history. Shares with their Clopper-Pearson 95 % intervals; a rejection is a p-value under 0.05.

The pilot: the seeds the design was set on. No gate reads it.

| channel | truth rejected | covers 1 | observational rejected | covers its factor | floor holds | failed | unchecked |
|---|---|---|---|---|---|---|---|
| pla | 0.030 [0.006, 0.085] | 0.970 [0.915, 0.994] | 1.000 [0.964, 1.000] | 0.959 [0.899, 0.989] | 0.980 [0.928, 0.998] | 0 | 0 |
| meta | 0.020 [0.002, 0.070] | 0.960 [0.901, 0.989] | 0.990 [0.946, 1.000] | 0.940 [0.874, 0.978] | 0.980 [0.930, 0.998] | 0 | 0 |
| tv | 0.010 [0.000, 0.054] | 0.950 [0.887, 0.984] | 0.960 [0.901, 0.989] | 0.800 [0.444, 0.975] | 0.900 [0.555, 0.997] | 0 | 0 |

The observational channel: the median factor the tests read of it, the median factor without noise, and its least `Gamma` at an effect-scale gap of 1, median and the share infinite (no level reconciles it with the tests); over the channels that predict a gap, and the count that predict none.

| channel | factor | noise-free | least Gamma | infinite | no factor |
|---|---|---|---|---|---|
| pla | 0.513 | 0.522 | 2.67 | 0.02 | 2 |
| meta | 0.202 | 0.194 | 7.02 | 0.00 | 0 |
| tv | 1.794 | 1.444 | 2.29 | 0.20 | 90 |
