# PyMC-Marketing 1.3.0's arm, pilot (float64)

The arm as `scripts/pymc_marketing_arm.py` fits it, static media and every lift test on the tool's default likelihood, in the environment `scripts/pymc_marketing_arm.txt` pins; every record names pymc-marketing 1.3.0, pymc 6.3.2, pytensor 3.3.6, arviz 1.3.0, numpy 2.4.6, scipy 1.18.1, pandas 3.1.0. Each world is the scorecard's version-2 export at rung 4, four go-dark tests a channel, every test whichever way its sales moved.

The pilot: the seeds the design was set on, apart from every scored one. No gate reads it.

Exported and scored under causal-hybrid-control 1.2.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.5.4, scipy 1.18.1: an export's bits, and so the digest a record echoes, move with them.

## The lift tests

*Tests*: the tests the exports hold, one row each. *Not fallen*: those whose sales did not fall, which PyMC-Marketing 1.2.0 refused and the budgets run's version-1 export left out. *Rows taken* and *refused*: the rows the arm gave the likelihood and those it did not, its records' `lift_rows` and `lift_dropped`; the geo model takes a test as one row a tested geo, so there they outnumber the tests, and *tests refused* counts tests. *Raised*: worlds whose script raised, whose arm then played the status quo; their record holds no count of rows. *Divergent*: fits with a divergent transition; *r-hat*: fits whose largest r-hat is over 1.01.

| family | environment | pilot worlds | fitted | tests | not fallen | rows taken | refused | tests refused | raised | divergent | r-hat | optimiser failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | drawn | 40 | 40 | 480 | 29 | 480 | 0 | n/a | 0 | 4 | 0 | 0 |
| 0 | reference | 20 | 20 | 240 | 20 | 240 | 0 | n/a | 0 | 5 | 0 | 0 |
| 13 | priors | 100 | 100 | 1200 | 51 | 1200 | 0 | n/a | 0 | 9 | 0 | 0 |
| 8 | policy-0 | 100 | 9 | 108 | 20 | 540 | 0 | 0 | 0 | 7 | 1 | 0 |
| 8 | policy-0.5 | 100 | 9 | 108 | 18 | 540 | 0 | 0 | 0 | 8 | 1 | 0 |
| 10 | geos-3 | 100 | 9 | 108 | 27 | 144 | 0 | 0 | 1 | 8 | 1 | 0 |
| 10 | geos-8 | 100 | 8 | 96 | 16 | 384 | 0 | 0 | 0 | 7 | 2 | 0 |
| 10 | geos-50 | 100 | 8 | 96 | 18 | 2400 | 0 | 0 | 0 | 8 | 1 | 0 |

Not run: family 8's policy-0, 91 of its 100 pilot worlds; family 8's policy-0.5, 91 of its 100 pilot worlds; family 10's geos-3, 91 of its 100 pilot worlds; family 10's geos-8, 92 of its 100 pilot worlds; family 10's geos-50, 92 of its 100 pilot worlds. Each environment's cells below read the first worlds of its seeded order it holds whole.

## Family 0: Track M v2, pilot

By construction: The world's media are PyMC-Marketing's default class: its logistic saturation is the world's tanh curve, and its normalised geometric adstock at the kernel's length is the world's kernel.
By construction: Every arm is told the kernel's length.
By construction: A geo test is two universes at the market's scale, so its gap holds no noise between regions beyond the noise drawn.

### k = 4, drawn

40 worlds, the first 40 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.1502 [+0.1182, +0.1822] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.1793 [+0.1252, +0.2334] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0410 [+0.0223, +0.0597] | 0 | 0 | +0.0971 [+0.0873, +0.1070] | +10.89 [+9.94, +11.84] | roi 0.733/0.875/0.933; mroi 0.833/0.892/0.967 | pull -0.025, miss 0.031 |

### k = 4, reference

20 worlds, the first 20 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.3373 [+0.3339, +0.3408] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.7288 [+0.7262, +0.7315] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0189 [-0.0048, +0.0427] | 0 | 0 | +0.0994 [+0.0844, +0.1143] | +11.25 [+9.83, +12.68] | roi 0.700/0.817/0.917; mroi 0.717/0.850/0.933 | pull +0.005, miss 0.030 |

### k = 4, pooled

40 worlds, the first 20 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.2438 [+0.2046, +0.2829] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.4512 [+0.3516, +0.5507] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0298 [+0.0127, +0.0468] | 0 | 0 | +0.0992 [+0.0889, +0.1095] | +11.29 [+10.33, +12.26] | roi 0.733/0.850/0.925; mroi 0.775/0.875/0.950 | pull -0.016, miss 0.031 |

## Family 13: PyMC-Marketing's home ground, pilot

By construction: The world's media are PyMC-Marketing's default class, by the bench's generator: each channel's spend over its largest week through PyMC-Marketing 1.2.0's normalised geometric adstock and its logistic saturation.
By construction: Alpha and lam are drawn from PyMC-Marketing's default priors, Beta(1, 3) and Gamma(3, 1), not from Track M v2's ranges; beta is not drawn from its prior but gives each channel the return over the history of the Track M v2 world beneath.
By construction: Every arm is told the kernel's length, Track M v2's.
By construction: Beneath the media each world is a Track M v2 drawn world: its base, its spend, its measurement layer and its quarter.
By construction: A geo test is two universes at the market's scale, so its gap holds no noise between regions beyond the noise drawn.

### k = 4, priors

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.2038 [+0.1790, +0.2286] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.1380 [+0.1100, +0.1659] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0907 [+0.0651, +0.1164] | 0 | 0 | +0.1013 [+0.0945, +0.1081] | +11.43 [+10.73, +12.13] | roi 0.740/0.820/0.893; mroi 0.740/0.833/0.903 | pull -0.057, miss 0.034 |

### k = 4, pooled

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.2038 [+0.1790, +0.2286] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.1380 [+0.1100, +0.1659] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0907 [+0.0651, +0.1164] | 0 | 0 | +0.1013 [+0.0945, +0.1081] | +11.43 [+10.73, +12.13] | roi 0.740/0.820/0.893; mroi 0.740/0.833/0.903 | pull -0.057, miss 0.034 |

## Family 8: geo, pilot

By construction: Each geo's effect on a channel is drawn from the hierarchy chc's geo model takes: log-normal about the channel's national median, independent across geos and channels.
By construction: A geo's lift is its population times a curve of its spend per head.
By construction: Every arm is told the kernel's length.
By construction: National media is allotted to the geos by population, so outside its tests every geo spends the same on it a head.

### k = 4, policy-0

9 worlds, the first 9 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.1119 [+0.0747, +0.1491] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0673 [+0.0392, +0.0954] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0258 [+0.0177, +0.0339] | 0 | 0 | +0.0062 [+0.0050, +0.0075] | +0.80 [+0.66, +0.95] | n/a | n/a |

### k = 4, policy-0.5

9 worlds, the first 9 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0824 [+0.0655, +0.0992] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0494 [+0.0321, +0.0667] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0258 [+0.0214, +0.0302] | 0 | 0 | +0.0076 [+0.0060, +0.0092] | +0.98 [+0.78, +1.18] | n/a | n/a |

### k = 4, pooled

18 worlds, the first 9 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0971 [+0.0775, +0.1168] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0583 [+0.0430, +0.0737] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0258 [+0.0217, +0.0299] | 0 | 0 | +0.0069 [+0.0059, +0.0079] | +0.89 [+0.77, +1.01] | n/a | n/a |

## Family 10: few geos, pilot

By construction: Each geo's effect on a channel is drawn from the hierarchy chc's geo model takes: log-normal about the channel's national median, independent across geos and channels.
By construction: A geo's lift is its population times a curve of its spend per head.
By construction: Every arm is told the kernel's length.
By construction: National media is allotted to the geos by population, so outside its tests every geo spends the same on it a head.

### k = 4, geos-3

9 worlds, the first 9 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0886 [+0.0545, +0.1227] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0649 [+0.0448, +0.0851] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0300 [+0.0008, +0.0592] | 1 | 0 | +0.0103 [+0.0058, +0.0149] | +1.38 [+0.76, +1.99] | n/a | n/a |

### k = 4, geos-8

8 worlds, the first 8 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0939 [+0.0674, +0.1205] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0652 [+0.0420, +0.0885] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0220 [+0.0117, +0.0323] | 0 | 0 | +0.0066 [+0.0049, +0.0082] | +0.84 [+0.66, +1.03] | n/a | n/a |

### k = 4, geos-50

8 worlds, the first 8 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0797 [+0.0673, +0.0920] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0533 [+0.0371, +0.0694] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0631 [+0.0499, +0.0763] | 0 | 0 | +0.0106 [+0.0085, +0.0127] | +1.35 [+1.08, +1.63] | n/a | n/a |

### k = 4, pooled

24 worlds, the first 8 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0890 [+0.0754, +0.1026] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0605 [+0.0501, +0.0708] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0385 [+0.0255, +0.0516] | 1 | 0 | +0.0090 [+0.0073, +0.0107] | +1.17 [+0.94, +1.40] | n/a | n/a |
