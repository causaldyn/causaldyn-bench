# Track M v2, budgets, the external arms, pilot (float64)

Regret per euro of the quarter's budget, scored as `results/track_m2_budgets.md` scores it; CHC, PyMC-Marketing and the status quo are that run's, read for the same worlds. Mean with its 95 % Student-t interval, the median, the fits that raised (their arm then played the status quo) and the plans moved into the box.

The pilot: the seeds the design was set on, apart from every scored one. No reading rule reads it.

## drawn: 40 worlds from seed 900

| arm | worlds | mean [95 %] | median | failed | moved |
|---|---|---|---|---|---|
| CHC | 40 | 0.0418 [0.0223, 0.0613] | 0.0205 | - | - |
| PyMC-Marketing | 40 | 0.0392 [0.0216, 0.0568] | 0.0128 | - | - |
| status quo | 40 | 0.1502 [0.1182, 0.1822] | 0.1283 | - | - |
| Meridian | 40 | 0.0983 [0.0752, 0.1215] | 0.0802 | 0 | 40 |
| Robyn | 10 | 0.1177 [0.0276, 0.2078] | 0.0652 | 0 | 0 |

Each comparison: the first arm's regret less the second's over the worlds both planned, and the worlds where the first's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| first | second | worlds | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|---|---|
| CHC | Meridian | 40 | -0.0565 [-0.0868, -0.0263] | 29 | 0 | 11 | 0.72 [0.56, 0.85] |
| PyMC-Marketing | Meridian | 40 | -0.0591 [-0.0875, -0.0308] | 33 | 0 | 7 | 0.82 [0.67, 0.93] |
| CHC | Robyn | 10 | -0.0859 [-0.1859, +0.0141] | 7 | 1 | 2 | 0.70 [0.35, 0.93] |
| PyMC-Marketing | Robyn | 10 | -0.0640 [-0.1748, +0.0468] | 6 | 0 | 4 | 0.60 [0.26, 0.88] |
| Meridian | status quo | 40 | -0.0518 [-0.0779, -0.0258] | 30 | 0 | 10 | 0.75 [0.59, 0.87] |
| Robyn | status quo | 10 | +0.0026 [-0.0659, +0.0710] | 5 | 0 | 5 | 0.50 [0.19, 0.81] |
| Meridian | Robyn | 10 | -0.0413 [-0.1142, +0.0315] | 6 | 0 | 4 | 0.60 [0.26, 0.88] |

Channels planned on a bound of the box (within 1% of its width): on Meridian's 40 worlds, Meridian's 4 of 120 against the oracle's 66; on Robyn's 10 worlds, Robyn's 13 of 30 against the oracle's 17.

Meridian's fits: 0 raised; of the rest, 4 had divergent transitions, 0 an R-hat over 1.01, 0 did not pass Meridian's own review; 40 plans moved into the box, the furthest by 0.00015 of a week's budget.

Robyn's fits over the genre ranges: 0 raised; of the rest, its own convergence check passed NRMSE in 5, DECOMP.RSSD in 0 and MAPE in 0; the model came from the clusters in 10; its allocator stopped on an error in 0; no plan moved into the box.

## reference: 20 worlds from seed 900

| arm | worlds | mean [95 %] | median | failed | moved |
|---|---|---|---|---|---|
| CHC | 20 | 0.0485 [0.0011, 0.0959] | 0.0000 | - | - |
| PyMC-Marketing | 20 | 0.0335 [-0.0110, 0.0780] | -0.0000 | - | - |
| status quo | 20 | 0.3373 [0.3339, 0.3408] | 0.3398 | - | - |
| Meridian | 20 | 0.0881 [0.0652, 0.1110] | 0.0932 | 0 | 20 |

Each comparison: the first arm's regret less the second's over the worlds both planned, and the worlds where the first's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| first | second | worlds | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|---|---|
| CHC | Meridian | 20 | -0.0396 [-0.0888, +0.0095] | 17 | 0 | 3 | 0.85 [0.62, 0.97] |
| PyMC-Marketing | Meridian | 20 | -0.0546 [-0.0965, -0.0127] | 18 | 0 | 2 | 0.90 [0.68, 0.99] |
| Meridian | status quo | 20 | -0.2492 [-0.2712, -0.2272] | 20 | 0 | 0 | 1.00 [0.83, 1.00] |

Channels planned on a bound of the box (within 1% of its width): on Meridian's 20 worlds, Meridian's 19 of 60 against the oracle's 40.

Meridian's fits: 0 raised; of the rest, 1 had divergent transitions, 0 an R-hat over 1.01, 0 did not pass Meridian's own review; 20 plans moved into the box, the furthest by 0.00035 of a week's budget.

## Variance reduction

What each design removes of a comparison's variance a world, so of the worlds it needs for one precision: the share, with its 95 % percentile bootstrap interval over the worlds (10000 resamples). *Pairing*: the arms scored on the same worlds (common random numbers), against an unpaired comparison, `1 - var(a - b) / (var(a) + var(b))`; negative where the arms' regrets move against each other. *Control variates*: the paired difference adjusted by the world's covariates, the status quo's and the equal split's regrets, the best plan's return per euro and the budget's shadow price, `1 - var(adjusted) / var(raw)`, each world by the slope fitted on the other worlds of its environment (left out), and its raw value against the others' mean, so the share is what a slope frozen on this pilot removes from a world it was not fitted on; it is the saving where the covariates' mean is read on many more worlds than the arms are scored on. *Both*: one less the product of what each keeps. `sd` is the paired difference's.

### drawn

| first | second | worlds | sd | pairing [95 %] | control variates [95 %] | both [95 %] |
|---|---|---|---|---|---|---|
| CHC | Meridian | 40 | 0.0946 | +0.001 [-0.221, +0.250] | +0.417 [+0.044, +0.666] | +0.417 [+0.192, +0.633] |
| PyMC-Marketing | Meridian | 40 | 0.0887 | +0.048 [-0.148, +0.313] | +0.561 [+0.256, +0.745] | +0.582 [+0.380, +0.752] |
| CHC | Robyn | 10 | 0.1398 | -0.086 [-0.348, +0.185] | -1.428 [-3.439, +0.411] | -1.637 [-3.933, +0.369] |
| PyMC-Marketing | Robyn | 10 | 0.1549 | -0.242 [-0.620, +0.175] | -0.878 [-2.727, +0.673] | -1.331 [-3.653, +0.561] |
| Meridian | status quo | 40 | 0.0813 | +0.566 [+0.252, +0.814] | +0.706 [+0.495, +0.802] | +0.872 [+0.801, +0.935] |
| Robyn | status quo | 10 | 0.0957 | +0.644 [-0.036, +0.908] | -1.433 [-7.731, +0.604] | +0.135 [-0.423, +0.771] |
| Meridian | Robyn | 10 | 0.1018 | +0.388 [+0.141, +0.538] | -1.493 [-4.565, +0.670] | -0.526 [-1.837, +0.749] |

### reference

| first | second | worlds | sd | pairing [95 %] | control variates [95 %] | both [95 %] |
|---|---|---|---|---|---|---|
| CHC | Meridian | 20 | 0.1050 | +0.128 [-0.428, +0.521] | -0.306 [-1.145, -0.066] | -0.139 [-1.297, +0.227] |
| PyMC-Marketing | Meridian | 20 | 0.0896 | +0.298 [+0.000, +0.713] | -0.210 [-2.031, +0.077] | +0.150 [-1.434, +0.550] |
| Meridian | status quo | 20 | 0.0470 | +0.100 [+0.012, +0.232] | -0.462 [-0.860, -0.130] | -0.316 [-0.675, +0.064] |
