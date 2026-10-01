# Track M v2, families (float64)

Regret per euro of the quarter's budget, on worlds whose channels' curve is each family in turn, every family on the same seeds. Mean with its 95 % Student-t interval.

| curve | status quo | tanh | AIC | robust | true family |
|---|---|---|---|---|---|
| tanh | 0.2027 [0.1783, 0.2270] | 0.0416 [0.0286, 0.0547] | 0.0732 [0.0517, 0.0948] | 0.1125 [0.0775, 0.1476] | 0.0416 [0.0286, 0.0547] |
| exponential | 0.1724 [0.1480, 0.1968] | 0.0584 [0.0409, 0.0758] | 0.0887 [0.0657, 0.1116] | 0.1095 [0.0794, 0.1396] | 0.0591 [0.0409, 0.0774] |
| michaelis-menten | 0.1723 [0.1473, 0.1973] | 0.0761 [0.0529, 0.0993] | 0.1109 [0.0818, 0.1400] | 0.1218 [0.0859, 0.1576] | 0.0864 [0.0608, 0.1120] |
| hill-2 | 0.4662 [0.4347, 0.4977] | 0.1258 [0.0903, 0.1613] | 0.2059 [0.1566, 0.2553] | 0.2162 [0.1679, 0.2645] | 0.2941 [0.2448, 0.3434] |
| weibull-2 | 0.6646 [0.6279, 0.7014] | 0.2292 [0.1792, 0.2791] | 0.3017 [0.2420, 0.3615] | 0.3280 [0.2688, 0.3872] | 0.5958 [0.5426, 0.6490] |
| logistic-4 | 0.9409 [0.8822, 0.9995] | 0.5833 [0.4926, 0.6739] | 0.3697 [0.2915, 0.4480] | 0.3885 [0.3075, 0.4694] | 0.9034 [0.8399, 0.9668] |

The worst family's mean: robust 0.3885 (logistic-4), AIC 0.3697 (logistic-4); robust less AIC +0.0187 [-0.0086, +0.0547] (bootstrap, 10000 draws of the seeds).

**Gate** (the robust plan's worst family mean regret below AIC's, the interval under nought): NOT met.

## tanh: 100 worlds from seed 30000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.2027 [0.1783, 0.2270] | 0.1812 | 0 |
| tanh | 0.0416 [0.0286, 0.0547] | 0.0045 | 0 |
| AIC | 0.0732 [0.0517, 0.0948] | 0.0250 | 0 |
| robust | 0.1125 [0.0775, 0.1476] | 0.0372 | 0 |
| true family | 0.0416 [0.0286, 0.0547] | 0.0045 | 0 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.0902 [-0.1289, -0.0514] | 76 | 0 | 24 | 0.76 [0.66, 0.84] |
| tanh | +0.0709 [+0.0362, +0.1055] | 26 | 22 | 52 | 0.26 [0.18, 0.36] |
| AIC | +0.0393 [+0.0089, +0.0696] | 28 | 27 | 45 | 0.28 [0.19, 0.38] |
| true family | +0.0709 [+0.0362, +0.1055] | 26 | 22 | 52 | 0.26 [0.18, 0.36] |

Of 300 channels, AIC picked the world's family (tanh) in 90, and the keep rule kept it in 204; families kept a channel: mean 4.06, distinct readings 3.15; the robust plan's readings, median 20, most 144. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 10, weibull 58, logistic 117; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## exponential: 100 worlds from seed 30000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.1724 [0.1480, 0.1968] | 0.1381 | 0 |
| tanh | 0.0584 [0.0409, 0.0758] | 0.0200 | 0 |
| AIC | 0.0887 [0.0657, 0.1116] | 0.0459 | 0 |
| robust | 0.1095 [0.0794, 0.1396] | 0.0662 | 0 |
| true family | 0.0591 [0.0409, 0.0774] | 0.0173 | 0 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.0629 [-0.1025, -0.0232] | 67 | 0 | 33 | 0.67 [0.57, 0.76] |
| tanh | +0.0512 [+0.0223, +0.0800] | 32 | 16 | 52 | 0.32 [0.23, 0.42] |
| AIC | +0.0208 [-0.0052, +0.0469] | 46 | 20 | 34 | 0.46 [0.36, 0.56] |
| true family | +0.0504 [+0.0215, +0.0793] | 29 | 15 | 56 | 0.29 [0.20, 0.39] |

Of 300 channels, AIC picked the world's family (exponential) in 20, and the keep rule kept it in 216; families kept a channel: mean 4.34, distinct readings 3.64; the robust plan's readings, median 30, most 180. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 6, weibull 68, logistic 86; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## michaelis-menten: 100 worlds from seed 30000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.1723 [0.1473, 0.1973] | 0.1490 | 0 |
| tanh | 0.0761 [0.0529, 0.0993] | 0.0284 | 0 |
| AIC | 0.1109 [0.0818, 0.1400] | 0.0470 | 0 |
| robust | 0.1218 [0.0859, 0.1576] | 0.0491 | 0 |
| true family | 0.0864 [0.0608, 0.1120] | 0.0289 | 0 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.0506 [-0.0957, -0.0054] | 62 | 0 | 38 | 0.62 [0.52, 0.72] |
| tanh | +0.0456 [+0.0149, +0.0764] | 38 | 7 | 55 | 0.38 [0.28, 0.48] |
| AIC | +0.0109 [-0.0175, +0.0393] | 40 | 13 | 47 | 0.40 [0.30, 0.50] |
| true family | +0.0353 [+0.0038, +0.0669] | 35 | 8 | 57 | 0.35 [0.26, 0.45] |

Of 300 channels, AIC picked the world's family (michaelis-menten) in 72, and the keep rule kept it in 226; families kept a channel: mean 4.70, distinct readings 4.08; the robust plan's readings, median 48, most 180. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 8, weibull 55, logistic 38; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## hill-2: 100 worlds from seed 30000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.4662 [0.4347, 0.4977] | 0.4498 | 0 |
| tanh | 0.1258 [0.0903, 0.1613] | 0.0014 | 0 |
| AIC | 0.2059 [0.1566, 0.2553] | 0.1246 | 0 |
| robust | 0.2162 [0.1679, 0.2645] | 0.1462 | 0 |
| true family | 0.2941 [0.2448, 0.3434] | 0.2707 | 29 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.2500 [-0.3026, -0.1974] | 80 | 0 | 20 | 0.80 [0.71, 0.87] |
| tanh | +0.0904 [+0.0397, +0.1410] | 16 | 39 | 45 | 0.16 [0.09, 0.25] |
| AIC | +0.0102 [-0.0154, +0.0359] | 20 | 60 | 20 | 0.20 [0.13, 0.29] |
| true family | -0.0779 [-0.1165, -0.0394] | 34 | 41 | 25 | 0.34 [0.25, 0.44] |

Of 300 channels, AIC picked the world's family (hill) in 142, and the keep rule kept it in 265; families kept a channel: mean 3.03, distinct readings 2.25; the robust plan's readings, median 9, most 48. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 30, weibull 79, logistic 197; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## weibull-2: 100 worlds from seed 30000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.6646 [0.6279, 0.7014] | 0.6509 | 0 |
| tanh | 0.2292 [0.1792, 0.2791] | 0.1329 | 0 |
| AIC | 0.3017 [0.2420, 0.3615] | 0.2957 | 0 |
| robust | 0.3280 [0.2688, 0.3872] | 0.3309 | 0 |
| true family | 0.5958 [0.5426, 0.6490] | 0.6159 | 76 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.3366 [-0.4042, -0.2691] | 86 | 0 | 14 | 0.86 [0.78, 0.92] |
| tanh | +0.0988 [+0.0317, +0.1660] | 18 | 40 | 42 | 0.18 [0.11, 0.27] |
| AIC | +0.0263 [+0.0062, +0.0463] | 10 | 71 | 19 | 0.10 [0.05, 0.18] |
| true family | -0.2678 [-0.3388, -0.1969] | 72 | 13 | 15 | 0.72 [0.62, 0.81] |

Of 300 channels, AIC picked the world's family (weibull) in 77, and the keep rule kept it in 202; families kept a channel: mean 2.65, distinct readings 1.95; the robust plan's readings, median 6, most 36. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 50, weibull 97, logistic 220; fits with a negative coefficient, planned as nought: tanh 1, exponential 1, michaelis-menten 1, hill 0, weibull 0, logistic 0.

## logistic-4: 100 worlds from seed 30000

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.9409 [0.8822, 0.9995] | 0.9296 | 0 |
| tanh | 0.5833 [0.4926, 0.6739] | 0.6290 | 0 |
| AIC | 0.3697 [0.2915, 0.4480] | 0.2729 | 0 |
| robust | 0.3885 [0.3075, 0.4694] | 0.2946 | 0 |
| true family | 0.9034 [0.8399, 0.9668] | 0.9184 | 94 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.5524 [-0.6494, -0.4554] | 92 | 0 | 8 | 0.92 [0.85, 0.96] |
| tanh | -0.1948 [-0.2879, -0.1017] | 36 | 51 | 13 | 0.36 [0.27, 0.46] |
| AIC | +0.0187 [-0.0147, +0.0521] | 6 | 86 | 8 | 0.06 [0.02, 0.13] |
| true family | -0.5149 [-0.6103, -0.4195] | 87 | 5 | 8 | 0.87 [0.79, 0.93] |

Of 300 channels, AIC picked the world's family (logistic) in 39, and the keep rule kept it in 80; families kept a channel: mean 2.44, distinct readings 1.75; the robust plan's readings, median 3, most 45. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 3, hill 92, weibull 148, logistic 220; fits with a negative coefficient, planned as nought: tanh 2, exponential 2, michaelis-menten 4, hill 0, weibull 0, logistic 0.
