# Track M v2, budgets, pilot (float64)

Regret per euro of the quarter's budget: what the best plan in the box returns over each arm's, on the world's own channels, carryover included. Mean with its 95 % Student-t interval, the median, and the fits that raised, whose arm then played the status quo.

The pilot: the seeds the design was set on, apart from every scored one. No gate reads it.

## drawn: 40 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.1502 [0.1182, 0.1822] | 0.1283 | 0 |
| equal split | 0.1793 [0.1252, 0.2334] | 0.1283 | 0 |
| observational | 0.2177 [0.1453, 0.2902] | 0.1637 | 0 |
| myopic | 0.0932 [0.0564, 0.1300] | 0.0423 | 0 |
| CHC | 0.0418 [0.0223, 0.0613] | 0.0205 | 0 |
| PyMC-Marketing | 0.0392 [0.0216, 0.0568] | 0.0128 | 0 |

CHC against each arm: CHC's regret minus the arm's over the same worlds, and the worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.1084 [-0.1485, -0.0683] | 34 | 0 | 6 | 0.85 [0.70, 0.94] |
| equal split | -0.1375 [-0.1967, -0.0782] | 31 | 0 | 9 | 0.78 [0.62, 0.89] |
| observational | -0.1759 [-0.2530, -0.0988] | 27 | 4 | 9 | 0.68 [0.51, 0.81] |
| myopic | -0.0514 [-0.0939, -0.0089] | 22 | 6 | 12 | 0.55 [0.38, 0.71] |
| PyMC-Marketing | +0.0026 [-0.0134, +0.0186] | 16 | 9 | 15 | 0.40 [0.25, 0.57] |

PyMC-Marketing's fits: 0 raised; of the rest, 2 had divergent transitions, 0 an r-hat over 1.01, 0 an optimiser that reported failure; 29 of 480 geo tests dropped, their sales not having fallen; 0 plans moved into the box.

CHC's readings: 108 of 120 channels' scale intervals open above, the tests not having read the curve's bend; the plan reads the family's shape past the tested spend.

## reference: 20 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.3373 [0.3339, 0.3408] | 0.3398 | 0 |
| equal split | 0.7288 [0.7262, 0.7315] | 0.7277 | 0 |
| observational | 0.3049 [0.1831, 0.4267] | 0.2545 | 0 |
| myopic | -0.0000 [-0.0000, 0.0000] | -0.0000 | 0 |
| CHC | 0.0485 [0.0011, 0.0959] | 0.0000 | 0 |
| PyMC-Marketing | 0.0335 [-0.0110, 0.0780] | -0.0000 | 0 |

CHC against each arm: CHC's regret minus the arm's over the same worlds, and the worlds where CHC's was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.2888 [-0.3365, -0.2412] | 19 | 0 | 1 | 0.95 [0.75, 1.00] |
| equal split | -0.6803 [-0.7273, -0.6334] | 20 | 0 | 0 | 1.00 [0.83, 1.00] |
| observational | -0.2564 [-0.3782, -0.1346] | 16 | 2 | 2 | 0.80 [0.56, 0.94] |
| myopic | +0.0485 [+0.0011, +0.0959] | 0 | 13 | 7 | 0.00 [0.00, 0.17] |
| PyMC-Marketing | +0.0150 [-0.0086, +0.0386] | 4 | 10 | 6 | 0.20 [0.06, 0.44] |

PyMC-Marketing's fits: 0 raised; of the rest, 1 had divergent transitions, 1 an r-hat over 1.01, 0 an optimiser that reported failure; 20 of 240 geo tests dropped, their sales not having fallen; 0 plans moved into the box.

CHC's readings: 51 of 60 channels' scale intervals open above, the tests not having read the curve's bend; the plan reads the family's shape past the tested spend.
