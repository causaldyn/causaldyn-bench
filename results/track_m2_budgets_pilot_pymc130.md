# Track M v2, budgets, pilot (float64)

Regret per euro of the quarter's budget: what the best plan in the box returns over each arm's, on the world's own channels, carryover included. Mean with its 95 % Student-t interval, the median, and the fits that raised, whose arm then played the status quo.

The pilot: the seeds the design was set on, apart from every scored one. No gate reads it.

## drawn: 40 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.1502 [0.1182, 0.1822] | 0.1283 | 0 |
| equal split | 0.1793 [0.1252, 0.2334] | 0.1283 | 0 |
| observational | 0.2175 [0.1451, 0.2899] | 0.1637 | 0 |
| myopic | 0.0932 [0.0564, 0.1300] | 0.0423 | 0 |
| CHC | 0.0418 [0.0223, 0.0613] | 0.0205 | 0 |
| PyMC-Marketing | 0.0426 [0.0241, 0.0611] | 0.0177 | 0 |

CHC against each arm: CHC's regret minus the arm's over the same worlds, and the worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.1084 [-0.1485, -0.0683] | 34 | 0 | 6 | 0.85 [0.70, 0.94] |
| equal split | -0.1375 [-0.1967, -0.0782] | 31 | 0 | 9 | 0.78 [0.62, 0.89] |
| observational | -0.1757 [-0.2527, -0.0987] | 27 | 5 | 8 | 0.68 [0.51, 0.81] |
| myopic | -0.0514 [-0.0939, -0.0089] | 22 | 6 | 12 | 0.55 [0.38, 0.71] |
| PyMC-Marketing | -0.0008 [-0.0166, +0.0150] | 16 | 9 | 15 | 0.40 [0.25, 0.57] |

PyMC-Marketing's fits: 0 raised; of the rest, 4 had divergent transitions, 0 an r-hat over 1.01, 0 an optimiser that reported failure; 29 of 480 geo tests dropped, their sales not having fallen; 0 plans moved into the box.

CHC's readings: 108 of 120 channels' scale intervals open above, the tests not having read the curve's bend; the plan reads the family's shape past the tested spend.

## reference: 20 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.3373 [0.3339, 0.3408] | 0.3398 | 0 |
| equal split | 0.7288 [0.7262, 0.7315] | 0.7277 | 0 |
| observational | 0.3049 [0.1831, 0.4267] | 0.2545 | 0 |
| myopic | -0.0000 [-0.0000, 0.0000] | 0.0000 | 0 |
| CHC | 0.0485 [0.0011, 0.0959] | 0.0000 | 0 |
| PyMC-Marketing | 0.0180 [-0.0055, 0.0414] | -0.0000 | 0 |

CHC against each arm: CHC's regret minus the arm's over the same worlds, and the worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.2888 [-0.3365, -0.2412] | 19 | 0 | 1 | 0.95 [0.75, 1.00] |
| equal split | -0.6803 [-0.7273, -0.6334] | 20 | 0 | 0 | 1.00 [0.83, 1.00] |
| observational | -0.2564 [-0.3782, -0.1346] | 16 | 2 | 2 | 0.80 [0.56, 0.94] |
| myopic | +0.0485 [+0.0011, +0.0959] | 0 | 13 | 7 | 0.00 [0.00, 0.17] |
| PyMC-Marketing | +0.0305 [-0.0146, +0.0757] | 3 | 11 | 6 | 0.15 [0.03, 0.38] |

PyMC-Marketing's fits: 0 raised; of the rest, 4 had divergent transitions, 0 an r-hat over 1.01, 0 an optimiser that reported failure; 20 of 240 geo tests dropped, their sales not having fallen; 0 plans moved into the box.

CHC's readings: 51 of 60 channels' scale intervals open above, the tests not having read the curve's bend; the plan reads the family's shape past the tested spend.

## Variance reduction

What each design removes of a comparison's variance a world, so of the worlds it needs for one precision: the share, with its 95 % percentile bootstrap interval over the worlds (10000 resamples). *Pairing*: the arms scored on the same worlds (common random numbers), against an unpaired comparison, `1 - var(a - b) / (var(a) + var(b))`; negative where the arms' regrets move against each other. *Control variates*: the paired difference adjusted by the world's covariates, the status quo's and the equal split's regrets, the best plan's return per euro and the budget's shadow price, `1 - var(adjusted) / var(raw)`, each world by the slope fitted on the other worlds of its environment (left out), and its raw value against the others' mean, so the share is what a slope frozen on this pilot removes from a world it was not fitted on; it is the saving where the covariates' mean is read on many more worlds than the arms are scored on. *Both*: one less the product of what each keeps. `sd` is the paired difference's.

### drawn

| first | second | worlds | sd | pairing [95 %] | control variates [95 %] | both [95 %] |
|---|---|---|---|---|---|---|
| CHC | status quo | 40 | 0.1253 | -0.145 [-0.326, +0.078] | +0.735 [+0.484, +0.885] | +0.696 [+0.496, +0.861] |
| CHC | equal split | 40 | 0.1853 | -0.063 [-0.181, +0.091] | +0.879 [+0.744, +0.953] | +0.871 [+0.737, +0.951] |
| CHC | observational | 40 | 0.2409 | -0.056 [-0.195, +0.168] | -0.054 [-0.754, +0.334] | -0.113 [-0.814, +0.320] |
| CHC | myopic | 40 | 0.1329 | -0.043 [-0.254, +0.227] | +0.067 [-0.439, +0.412] | +0.027 [-0.521, +0.425] |
| CHC | PyMC-Marketing | 40 | 0.0494 | +0.654 [+0.370, +0.823] | -0.016 [-0.508, +0.195] | +0.649 [+0.279, +0.811] |

### reference

| first | second | worlds | sd | pairing [95 %] | control variates [95 %] | both [95 %] |
|---|---|---|---|---|---|---|
| CHC | status quo | 20 | 0.1018 | -0.005 [-0.136, +0.029] | -0.171 [-1.554, +0.148] | -0.176 [-1.839, +0.134] |
| CHC | equal split | 20 | 0.1003 | +0.023 [-0.085, +0.045] | -0.207 [-1.664, +0.112] | -0.179 [-1.902, +0.133] |
| CHC | observational | 20 | 0.2602 | +0.132 [-0.174, +0.474] | +0.637 [+0.370, +0.819] | +0.685 [+0.454, +0.853] |
| CHC | myopic | 20 | 0.1013 | +0.000 [-0.000, +0.000] | -0.183 [-1.751, +0.130] | -0.183 [-1.748, +0.130] |
| CHC | PyMC-Marketing | 20 | 0.0964 | +0.273 [-0.137, +0.808] | -0.497 [-1.484, -0.259] | -0.089 [-1.157, +0.559] |
