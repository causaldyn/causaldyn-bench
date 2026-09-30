# Track M v2, budgets (float64)

Regret per euro of the quarter's budget: what the best plan in the box returns over each arm's, on the world's own channels, carryover included. Mean with its 95 % Student-t interval, the median, and the fits that raised, whose arm then played the status quo.

## drawn: 200 worlds from seed 10000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.2091 [0.1893, 0.2289] | 0.1842 | 0 |
| equal split | 0.2180 [0.1906, 0.2454] | 0.1468 | 0 |
| observational | 0.2197 [0.1829, 0.2564] | 0.1325 | 0 |
| myopic | 0.0750 [0.0588, 0.0912] | 0.0195 | 0 |
| CHC | 0.0497 [0.0376, 0.0619] | 0.0116 | 0 |
| PyMC-Marketing | 0.0477 [0.0379, 0.0576] | 0.0170 | 0 |

CHC against each arm: CHC's regret minus the arm's over the same worlds, and the worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.1594 [-0.1821, -0.1367] | 170 | 0 | 30 | 0.85 [0.79, 0.90] |
| equal split | -0.1682 [-0.1977, -0.1388] | 162 | 0 | 38 | 0.81 [0.75, 0.86] |
| observational | -0.1699 [-0.2096, -0.1303] | 135 | 17 | 48 | 0.68 [0.61, 0.74] |
| myopic | -0.0253 [-0.0420, -0.0086] | 91 | 49 | 60 | 0.46 [0.38, 0.53] |
| PyMC-Marketing | +0.0020 [-0.0110, +0.0150] | 79 | 43 | 78 | 0.40 [0.33, 0.47] |

PyMC-Marketing's fits: 0 raised; of the rest, 15 had divergent transitions, 4 an r-hat over 1.01, 0 an optimiser that reported failure; 127 of 2400 geo tests dropped, their sales not having fallen; 0 plans moved into the box.

CHC's readings: 554 of 600 channels' scale intervals open above, the tests not having read the curve's bend; the plan reads the family's shape past the tested spend.

**Gate** (CHC's regret below PyMC-Marketing's on the drawn worlds, the paired difference's 95 % interval under nought): NOT met.

## reference: 100 worlds from seed 20000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.3365 [0.3348, 0.3382] | 0.3365 | 0 |
| equal split | 0.7271 [0.7258, 0.7284] | 0.7277 | 0 |
| observational | 0.3355 [0.2828, 0.3883] | 0.2853 | 0 |
| myopic | 0.0031 [0.0007, 0.0056] | 0.0000 | 0 |
| CHC | 0.0278 [0.0148, 0.0408] | 0.0000 | 0 |
| PyMC-Marketing | 0.0115 [0.0040, 0.0189] | 0.0000 | 0 |

CHC against each arm: CHC's regret minus the arm's over the same worlds, and the worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.3087 [-0.3218, -0.2956] | 99 | 0 | 1 | 0.99 [0.95, 1.00] |
| equal split | -0.6993 [-0.7122, -0.6865] | 100 | 0 | 0 | 1.00 [0.96, 1.00] |
| observational | -0.3077 [-0.3631, -0.2524] | 82 | 10 | 8 | 0.82 [0.73, 0.89] |
| myopic | +0.0247 [+0.0118, +0.0376] | 2 | 67 | 31 | 0.02 [0.00, 0.07] |
| PyMC-Marketing | +0.0163 [+0.0032, +0.0295] | 12 | 59 | 29 | 0.12 [0.06, 0.20] |

PyMC-Marketing's fits: 0 raised; of the rest, 12 had divergent transitions, 2 an r-hat over 1.01, 0 an optimiser that reported failure; 88 of 1200 geo tests dropped, their sales not having fallen; 0 plans moved into the box.

CHC's readings: 266 of 300 channels' scale intervals open above, the tests not having read the curve's bend; the plan reads the family's shape past the tested spend.
