# Track M v2, budgets, the external arms (float64)

Regret per euro of the quarter's budget, scored as `results/track_m2_budgets.md` scores it; CHC, PyMC-Marketing and the status quo are that run's, read for the same worlds. Mean with its 95 % Student-t interval, the median, the fits that raised (their arm then played the status quo) and the plans moved into the box.

## drawn: 200 worlds from seed 10000

| arm | worlds | mean [95 %] | median | failed | moved |
|---|---|---|---|---|---|
| CHC | 200 | 0.0497 [0.0376, 0.0619] | 0.0116 | - | - |
| PyMC-Marketing | 200 | 0.0477 [0.0379, 0.0576] | 0.0170 | - | - |
| status quo | 200 | 0.2091 [0.1893, 0.2289] | 0.1842 | - | - |
| Meridian | 200 | 0.1195 [0.1080, 0.1310] | 0.1167 | 0 | 199 |
| Robyn | 100 | 0.2427 [0.1961, 0.2892] | 0.1844 | 0 | 0 |

Each comparison: the first arm's regret less the second's over the worlds both planned, and the worlds where the first's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| first | second | worlds | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|---|---|
| CHC | Meridian | 200 | -0.0697 [-0.0855, -0.0540] | 153 | 0 | 47 | 0.77 [0.70, 0.82] |
| PyMC-Marketing | Meridian | 200 | -0.0717 [-0.0852, -0.0583] | 145 | 0 | 55 | 0.72 [0.66, 0.79] |
| CHC | Robyn | 100 | -0.1815 [-0.2287, -0.1343] | 78 | 3 | 19 | 0.78 [0.69, 0.86] |
| PyMC-Marketing | Robyn | 100 | -0.1914 [-0.2381, -0.1447] | 77 | 4 | 19 | 0.77 [0.68, 0.85] |
| Meridian | status quo | 200 | -0.0897 [-0.1040, -0.0753] | 174 | 0 | 26 | 0.87 [0.82, 0.91] |
| Robyn | status quo | 100 | +0.0458 [+0.0115, +0.0801] | 45 | 0 | 55 | 0.45 [0.35, 0.55] |
| Meridian | Robyn | 100 | -0.1273 [-0.1656, -0.0891] | 70 | 0 | 30 | 0.70 [0.60, 0.79] |

Channels planned on a bound of the box (within 1% of its width): on Meridian's 200 worlds, Meridian's 24 of 600 against the oracle's 347; on Robyn's 100 worlds, Robyn's 131 of 300 against the oracle's 170.

Meridian's fits: 0 raised; of the rest, 10 had divergent transitions, 0 an R-hat over 1.01, 0 did not pass Meridian's own review; 199 plans moved into the box, the furthest by 0.00032 of a week's budget.

Robyn's fits over the genre ranges: 0 raised; of the rest, its own convergence check passed NRMSE in 54, DECOMP.RSSD in 0 and MAPE in 0; the model came from the clusters in 100; its allocator stopped on an error in 0; no plan moved into the box.

**Reading** (on the drawn worlds, by the paired difference's 95 % interval): CHC beats Meridian; PyMC-Marketing beats Meridian; CHC beats Robyn; PyMC-Marketing beats Robyn.

## reference: 100 worlds from seed 20000

| arm | worlds | mean [95 %] | median | failed | moved |
|---|---|---|---|---|---|
| CHC | 100 | 0.0278 [0.0148, 0.0408] | 0.0000 | - | - |
| PyMC-Marketing | 100 | 0.0115 [0.0040, 0.0189] | 0.0000 | - | - |
| status quo | 100 | 0.3365 [0.3348, 0.3382] | 0.3365 | - | - |
| Meridian | 100 | 0.0868 [0.0785, 0.0952] | 0.0804 | 0 | 100 |

Each comparison: the first arm's regret less the second's over the worlds both planned, and the worlds where the first's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| first | second | worlds | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|---|---|
| CHC | Meridian | 100 | -0.0591 [-0.0730, -0.0452] | 91 | 0 | 9 | 0.91 [0.84, 0.96] |
| PyMC-Marketing | Meridian | 100 | -0.0754 [-0.0840, -0.0668] | 96 | 0 | 4 | 0.96 [0.90, 0.99] |
| Meridian | status quo | 100 | -0.2497 [-0.2584, -0.2409] | 100 | 0 | 0 | 1.00 [0.96, 1.00] |

Channels planned on a bound of the box (within 1% of its width): on Meridian's 100 worlds, Meridian's 100 of 300 against the oracle's 200.

Meridian's fits: 0 raised; of the rest, 3 had divergent transitions, 0 an R-hat over 1.01, 0 did not pass Meridian's own review; 100 plans moved into the box, the furthest by 0.00037 of a week's budget.
