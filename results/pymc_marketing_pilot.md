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
| 8 | policy-0 | 100 | 100 | 1200 | 177 | 5940 | 0 | 0 | 1 | 96 | 4 | 0 |
| 8 | policy-0.5 | 100 | 100 | 1200 | 215 | 5940 | 0 | 0 | 1 | 81 | 5 | 0 |
| 10 | geos-3 | 100 | 100 | 1200 | 281 | 1692 | 0 | 0 | 6 | 94 | 11 | 0 |
| 10 | geos-8 | 100 | 100 | 1200 | 222 | 4752 | 0 | 0 | 1 | 97 | 10 | 0 |
| 10 | geos-50 | 100 | 100 | 1200 | 253 | 30000 | 0 | 0 | 0 | 100 | 6 | 0 |

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

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0871 [+0.0788, +0.0955] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0626 [+0.0544, +0.0708] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0271 [+0.0248, +0.0293] | 1 | 0 | +0.0060 [+0.0057, +0.0063] | +0.76 [+0.72, +0.80] | n/a | n/a |

### k = 4, policy-0.5

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0966 [+0.0882, +0.1051] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0621 [+0.0551, +0.0690] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0293 [+0.0263, +0.0322] | 1 | 0 | +0.0073 [+0.0068, +0.0077] | +0.92 [+0.86, +0.97] | n/a | n/a |

### k = 4, pooled

200 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0919 [+0.0859, +0.0978] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0623 [+0.0570, +0.0677] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0282 [+0.0263, +0.0300] | 2 | 0 | +0.0066 [+0.0063, +0.0069] | +0.84 [+0.80, +0.87] | n/a | n/a |

## Family 10: few geos, pilot

By construction: Each geo's effect on a channel is drawn from the hierarchy chc's geo model takes: log-normal about the channel's national median, independent across geos and channels.
By construction: A geo's lift is its population times a curve of its spend per head.
By construction: Every arm is told the kernel's length.
By construction: National media is allotted to the geos by population, so outside its tests every geo spends the same on it a head.

### k = 4, geos-3

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0874 [+0.0748, +0.1001] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0582 [+0.0506, +0.0658] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0276 [+0.0213, +0.0339] | 6 | 0 | +0.0101 [+0.0093, +0.0108] | +1.28 [+1.19, +1.38] | n/a | n/a |

### k = 4, geos-8

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.1005 [+0.0900, +0.1110] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0657 [+0.0577, +0.0737] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0274 [+0.0241, +0.0306] | 1 | 0 | +0.0067 [+0.0063, +0.0071] | +0.85 [+0.80, +0.90] | n/a | n/a |

### k = 4, geos-50

100 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0910 [+0.0858, +0.0963] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0576 [+0.0539, +0.0613] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0607 [+0.0584, +0.0630] | 0 | 0 | +0.0100 [+0.0095, +0.0104] | +1.27 [+1.21, +1.33] | n/a | n/a |

### k = 4, pooled

300 worlds, the first 100 of each environment.

| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 | effect against spend share |
|---|---|---|---|---|---|---|---|
| status quo | +0.0930 [+0.0873, +0.0987] | 0 | 0 | n/a | n/a | n/a | n/a |
| equal split | +0.0605 [+0.0567, +0.0643] | 0 | 0 | n/a | n/a | n/a | n/a |
| pymc-marketing | +0.0386 [+0.0355, +0.0416] | 7 | 0 | +0.0089 [+0.0085, +0.0093] | +1.13 [+1.09, +1.18] | n/a | n/a |
