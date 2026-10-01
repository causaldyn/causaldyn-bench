# Track M v2, geo selection (500 worlds from seed 50000, float64)

Regret per euro of the quarter's budget of the plan each arm's test leads to, paid shopping fitted from the test and the other channels known. Mean with its 95 % Student-t interval, and the median.

| arm | mean [95 %] | median | spend per head | screen | placebo | line | failed |
|---|---|---|---|---|---|---|---|
| regret | 0.0050 [0.0037, 0.0062] | 0.0000 | 1.49 | 34.1 | 38.4 | 52 | 0 |
| Abadie-Zhao | 0.0143 [0.0111, 0.0176] | 0.0002 | 1.01 | 38.4 | 32.8 | 139 | 0 |
| least RMSPE | 0.0164 [0.0131, 0.0197] | 0.0003 | 0.98 | 26.2 | 34.8 | 143 | 0 |
| random | 0.0241 [0.0187, 0.0295] | 0.0007 | 0.98 | 43.4 | 37.5 | 142 | 0 |
| heaviest | 0.0289 [0.0224, 0.0355] | 0.0017 | 2.05 | 72.8 | 62.0 | 28 | 0 |

Spend per head, screen and placebo are the chosen sets' medians: paid shopping's spend per head over the market's, and the synthetic control's in-time errors per head on the two windows, the screen the one the choice reads. Line counts the worlds whose fit ran the scale past 1e+06, reading the curve as a line.

The regret arm against each: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] | same set |
|---|---|---|---|---|---|---|
| Abadie-Zhao | -0.0094 [-0.0128, -0.0059] | 182 | 222 | 96 | 0.36 [0.32, 0.41] | 0.01 |
| least RMSPE | -0.0114 [-0.0148, -0.0080] | 181 | 236 | 83 | 0.36 [0.32, 0.41] | 0.12 |
| random | -0.0191 [-0.0246, -0.0136] | 200 | 207 | 93 | 0.40 [0.36, 0.44] | 0.02 |
| heaviest | -0.0240 [-0.0304, -0.0175] | 212 | 237 | 51 | 0.42 [0.38, 0.47] | 0.16 |

The reading's error over the readouts: each world's share of weeks within 1.96 of the chosen set's placebo errors, its mean over the worlds with a Student-t interval, and the median of the error's RMS over the placebo and over the screen.

| arm | covered [95 %] | error / placebo | error / screen |
|---|---|---|---|
| regret | 0.936 [0.931, 0.942] | 0.98 | 1.11 |
| Abadie-Zhao | 0.928 [0.923, 0.934] | 1.03 | 0.88 |
| least RMSPE | 0.938 [0.933, 0.943] | 0.98 | 1.33 |
| random | 0.932 [0.927, 0.938] | 1.01 | 0.87 |
| heaviest | 0.923 [0.915, 0.931] | 1.02 | 0.90 |

The regret arm's coverage less the random arm's, paired by world: +0.004 [-0.004, +0.012]. The regret its sets' tests were expected to leave, at the placebo: 0.0028 [0.0025, 0.0030] per euro, against the 0.0050 they left.

Of the prior's draws, 0.56 hold paid shopping at an end of its box, where it weighs nothing; of the pool's sets, 0.00 leave the regret unbounded, their test unable to part the three parameters.

**Gate** (the regret arm's mean below Abadie-Zhao and random's, each interval under nought; its set another's in under 50% of worlds; its coverage at most 0.02 under the random arm's): met.
