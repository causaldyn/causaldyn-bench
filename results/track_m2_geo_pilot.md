# Track M v2, geo selection, pilot (40 worlds from seed 900, float64)

Regret per euro of the quarter's budget of the plan each arm's test leads to, paid shopping fitted from the test and the other channels known. Mean with its 95 % Student-t interval, and the median.

The pilot: the seeds the design was set on, apart from every scored one. No gate reads it.

| arm | mean [95 %] | median | spend per head | screen | placebo | line | failed |
|---|---|---|---|---|---|---|---|
| regret | 0.0079 [0.0024, 0.0134] | 0.0034 | 1.53 | 35.8 | 38.8 | 6 | 0 |
| Abadie-Zhao | 0.0153 [0.0072, 0.0234] | 0.0033 | 0.97 | 37.8 | 32.8 | 8 | 0 |
| least RMSPE | 0.0301 [0.0141, 0.0460] | 0.0050 | 0.98 | 26.8 | 36.3 | 14 | 0 |
| random | 0.0435 [0.0168, 0.0703] | 0.0133 | 0.85 | 42.1 | 36.5 | 14 | 0 |
| heaviest | 0.0535 [0.0171, 0.0898] | 0.0092 | 2.16 | 83.1 | 77.3 | 3 | 0 |

Spend per head, screen and placebo are the chosen sets' medians: paid shopping's spend per head over the market's, and the synthetic control's in-time errors per head on the two windows, the screen the one the choice reads. Line counts the worlds whose fit ran the scale past 1e+06, reading the curve as a line.

The regret arm against each: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] | same set |
|---|---|---|---|---|---|---|
| Abadie-Zhao | -0.0074 [-0.0173, +0.0025] | 17 | 12 | 11 | 0.42 [0.27, 0.59] | 0.03 |
| least RMSPE | -0.0222 [-0.0380, -0.0064] | 15 | 14 | 11 | 0.38 [0.23, 0.54] | 0.12 |
| random | -0.0356 [-0.0635, -0.0077] | 24 | 7 | 9 | 0.60 [0.43, 0.75] | 0.00 |
| heaviest | -0.0456 [-0.0829, -0.0083] | 20 | 12 | 8 | 0.50 [0.34, 0.66] | 0.10 |

The reading's error over the readouts: each world's share of weeks within 1.96 of the chosen set's placebo errors, its mean over the worlds with a Student-t interval, and the median of the error's RMS over the placebo and over the screen.

| arm | covered [95 %] | error / placebo | error / screen |
|---|---|---|---|
| regret | 0.963 [0.954, 0.972] | 0.97 | 1.04 |
| Abadie-Zhao | 0.928 [0.908, 0.948] | 1.03 | 0.89 |
| least RMSPE | 0.947 [0.933, 0.961] | 0.99 | 1.34 |
| random | 0.936 [0.917, 0.954] | 1.00 | 0.89 |
| heaviest | 0.937 [0.913, 0.961] | 1.00 | 1.00 |

The regret arm's coverage less the random arm's, paired by world: +0.028 [+0.008, +0.048]. The regret its sets' tests were expected to leave, at the placebo: 0.0047 [0.0026, 0.0067] per euro, against the 0.0079 they left.

Of the prior's draws, 0.54 hold paid shopping at an end of its box, where it weighs nothing; of the pool's sets, 0.00 leave the regret unbounded, their test unable to part the three parameters.
