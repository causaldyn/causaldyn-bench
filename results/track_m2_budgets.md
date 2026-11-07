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

## Variance reduction

What each design removes of a comparison's variance a world, so of the worlds it needs for one precision: the share, with its 95 % percentile bootstrap interval over the worlds (10000 resamples). *Pairing*: the arms scored on the same worlds (common random numbers), against an unpaired comparison, `1 - var(a - b) / (var(a) + var(b))`; negative where the arms' regrets move against each other. *Control variates*: the paired difference adjusted by the world's covariates, the status quo's and the equal split's regrets, the best plan's return per euro and the budget's shadow price, `1 - var(adjusted) / var(raw)`, each world by the slope fitted on the worlds of `track_m2_budgets_pilot`, the pilot of these arms in this environment, none of them a world here, and frozen; *on the pilot* is the share that pilot read on its own worlds, left out, the number this run checks out of sample; it is the saving where the covariates' mean is read on many more worlds than the arms are scored on. *Both*: one less the product of what each keeps. `sd` is the paired difference's.

### drawn

| first | second | worlds | sd | pairing [95 %] | control variates [95 %] | both [95 %] | on the pilot [95 %] |
|---|---|---|---|---|---|---|---|
| CHC | status quo | 200 | 0.1631 | +0.041 [-0.076, +0.163] | +0.688 [+0.513, +0.824] | +0.701 [+0.578, +0.816] | +0.735 [+0.484, +0.885] |
| CHC | equal split | 200 | 0.2110 | +0.034 [-0.060, +0.151] | +0.814 [+0.686, +0.900] | +0.820 [+0.727, +0.897] | +0.879 [+0.744, +0.953] |
| CHC | observational | 200 | 0.2843 | -0.049 [-0.120, +0.033] | -0.082 [-0.378, +0.124] | -0.135 [-0.444, +0.083] | -0.054 [-0.750, +0.332] |
| CHC | myopic | 200 | 0.1198 | +0.320 [+0.128, +0.526] | -0.413 [-0.877, -0.134] | +0.039 [-0.279, +0.281] | +0.067 [-0.439, +0.412] |
| CHC | PyMC-Marketing | 200 | 0.0935 | +0.303 [+0.089, +0.524] | -0.081 [-0.157, -0.017] | +0.247 [+0.025, +0.468] | -0.003 [-0.458, +0.194] |

### reference

| first | second | worlds | sd | pairing [95 %] | control variates [95 %] | both [95 %] | on the pilot [95 %] |
|---|---|---|---|---|---|---|---|
| CHC | status quo | 100 | 0.0661 | +0.007 [-0.036, +0.052] | -1.015 [-3.006, -0.429] | -1.001 [-2.940, -0.415] | -0.171 [-1.554, +0.148] |
| CHC | equal split | 100 | 0.0649 | +0.033 [-0.009, +0.077] | -1.085 [-3.114, -0.482] | -1.017 [-3.020, -0.422] | -0.207 [-1.664, +0.112] |
| CHC | observational | 100 | 0.2790 | -0.038 [-0.122, +0.039] | -0.875 [-1.343, -0.445] | -0.945 [-1.450, -0.482] | +0.637 [+0.370, +0.819] |
| CHC | myopic | 100 | 0.0651 | +0.052 [-0.006, +0.152] | -1.127 [-3.412, -0.501] | -1.016 [-2.936, -0.410] | -0.183 [-1.751, +0.130] |
| CHC | PyMC-Marketing | 100 | 0.0662 | +0.236 [-0.057, +0.607] | -0.155 [-0.410, -0.050] | +0.118 [-0.271, +0.498] | -0.306 [-1.225, -0.118] |
