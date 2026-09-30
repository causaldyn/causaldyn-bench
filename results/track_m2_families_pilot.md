# Track M v2, families, pilot (float64)

Regret per euro of the quarter's budget, on worlds whose channels' curve is each family in turn, every family on the same seeds. Mean with its 95 % Student-t interval.

The pilot: the seeds the design was set on, apart from every scored one. No gate reads it.

| curve | status quo | tanh | AIC | robust | true family |
|---|---|---|---|---|---|
| tanh | 0.1151 [0.0440, 0.1863] | 0.0318 [-0.0013, 0.0648] | 0.0197 [0.0047, 0.0347] | 0.0314 [-0.0082, 0.0711] | 0.0318 [-0.0013, 0.0648] |
| exponential | 0.0988 [0.0387, 0.1589] | 0.0325 [0.0039, 0.0610] | 0.0485 [0.0174, 0.0795] | 0.0664 [0.0036, 0.1293] | 0.0306 [0.0022, 0.0591] |
| michaelis-menten | 0.1218 [0.0653, 0.1782] | 0.0509 [-0.0063, 0.1081] | 0.0885 [0.0311, 0.1458] | 0.0729 [0.0119, 0.1340] | 0.0896 [0.0109, 0.1683] |
| hill-2 | 0.3773 [0.2766, 0.4781] | 0.1302 [0.0376, 0.2227] | 0.1112 [0.0187, 0.2037] | 0.0896 [0.0032, 0.1761] | 0.1412 [0.0293, 0.2530] |
| weibull-2 | 0.5474 [0.4201, 0.6746] | 0.4042 [0.2392, 0.5691] | 0.1997 [0.0500, 0.3495] | 0.2080 [-0.0216, 0.4375] | 0.3949 [0.1576, 0.6323] |
| logistic-4 | 0.7513 [0.5167, 0.9859] | 0.6941 [0.4516, 0.9365] | 0.3731 [0.0307, 0.7154] | 0.3564 [0.0063, 0.7064] | 0.7513 [0.5167, 0.9859] |

The worst family's mean: robust 0.3564 (logistic-4), AIC 0.3731 (logistic-4); robust less AIC -0.0167 [-0.0501, +0.0000] (bootstrap, 10000 draws of the seeds).

## tanh: 10 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.1151 [0.0440, 0.1863] | 0.1035 | 0 |
| tanh | 0.0318 [-0.0013, 0.0648] | 0.0205 | 0 |
| AIC | 0.0197 [0.0047, 0.0347] | 0.0168 | 0 |
| robust | 0.0314 [-0.0082, 0.0711] | 0.0075 | 0 |
| true family | 0.0318 [-0.0013, 0.0648] | 0.0205 | 0 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.0837 [-0.1692, +0.0018] | 9 | 0 | 1 | 0.90 [0.55, 1.00] |
| tanh | -0.0003 [-0.0526, +0.0520] | 3 | 4 | 3 | 0.30 [0.07, 0.65] |
| AIC | +0.0117 [-0.0257, +0.0492] | 2 | 5 | 3 | 0.20 [0.03, 0.56] |
| true family | -0.0003 [-0.0526, +0.0520] | 3 | 4 | 3 | 0.30 [0.07, 0.65] |

Of 30 channels, AIC picked the world's family (tanh) in 9, and the keep rule kept it in 20; families kept a channel: mean 3.93, distinct readings 3.30; the robust plan's readings, median 22, most 150. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 2, weibull 7, logistic 12; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## exponential: 10 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.0988 [0.0387, 0.1589] | 0.0610 | 0 |
| tanh | 0.0325 [0.0039, 0.0610] | 0.0129 | 0 |
| AIC | 0.0485 [0.0174, 0.0795] | 0.0410 | 0 |
| robust | 0.0664 [0.0036, 0.1293] | 0.0356 | 0 |
| true family | 0.0306 [0.0022, 0.0591] | 0.0115 | 0 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.0324 [-0.1071, +0.0423] | 7 | 0 | 3 | 0.70 [0.35, 0.93] |
| tanh | +0.0340 [-0.0316, +0.0996] | 2 | 3 | 5 | 0.20 [0.03, 0.56] |
| AIC | +0.0180 [-0.0506, +0.0865] | 4 | 3 | 3 | 0.40 [0.12, 0.74] |
| true family | +0.0358 [-0.0307, +0.1023] | 2 | 3 | 5 | 0.20 [0.03, 0.56] |

Of 30 channels, AIC picked the world's family (exponential) in 1, and the keep rule kept it in 23; families kept a channel: mean 4.73, distinct readings 4.13; the robust plan's readings, median 60, most 150. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 4, logistic 8; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## michaelis-menten: 10 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.1218 [0.0653, 0.1782] | 0.1016 | 0 |
| tanh | 0.0509 [-0.0063, 0.1081] | 0.0301 | 0 |
| AIC | 0.0885 [0.0311, 0.1458] | 0.0666 | 0 |
| robust | 0.0729 [0.0119, 0.1340] | 0.0329 | 0 |
| true family | 0.0896 [0.0109, 0.1683] | 0.0430 | 0 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.0488 [-0.1036, +0.0059] | 7 | 0 | 3 | 0.70 [0.35, 0.93] |
| tanh | +0.0220 [-0.0170, +0.0610] | 3 | 1 | 6 | 0.30 [0.07, 0.65] |
| AIC | -0.0155 [-0.0417, +0.0106] | 6 | 1 | 3 | 0.60 [0.26, 0.88] |
| true family | -0.0167 [-0.0882, +0.0549] | 3 | 1 | 6 | 0.30 [0.07, 0.65] |

Of 30 channels, AIC picked the world's family (michaelis-menten) in 8, and the keep rule kept it in 27; families kept a channel: mean 5.10, distinct readings 4.50; the robust plan's readings, median 114, most 125. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 1, weibull 9, logistic 4; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## hill-2: 10 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.3773 [0.2766, 0.4781] | 0.3421 | 0 |
| tanh | 0.1302 [0.0376, 0.2227] | 0.1030 | 0 |
| AIC | 0.1112 [0.0187, 0.2037] | 0.0606 | 0 |
| robust | 0.0896 [0.0032, 0.1761] | 0.0009 | 0 |
| true family | 0.1412 [0.0293, 0.2530] | 0.0991 | 2 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.2877 [-0.4440, -0.1314] | 10 | 0 | 0 | 1.00 [0.69, 1.00] |
| tanh | -0.0405 [-0.1100, +0.0289] | 5 | 3 | 2 | 0.50 [0.19, 0.81] |
| AIC | -0.0215 [-0.0860, +0.0429] | 3 | 4 | 3 | 0.30 [0.07, 0.65] |
| true family | -0.0515 [-0.1285, +0.0254] | 3 | 3 | 4 | 0.30 [0.07, 0.65] |

Of 30 channels, AIC picked the world's family (hill) in 11, and the keep rule kept it in 25; families kept a channel: mean 3.13, distinct readings 2.20; the robust plan's readings, median 8, most 24. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 2, weibull 8, logistic 19; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## weibull-2: 10 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.5474 [0.4201, 0.6746] | 0.5071 | 0 |
| tanh | 0.4042 [0.2392, 0.5691] | 0.4001 | 0 |
| AIC | 0.1997 [0.0500, 0.3495] | 0.1462 | 0 |
| robust | 0.2080 [-0.0216, 0.4375] | 0.0054 | 0 |
| true family | 0.3949 [0.1576, 0.6323] | 0.4345 | 6 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.3394 [-0.5773, -0.1016] | 9 | 0 | 1 | 0.90 [0.55, 1.00] |
| tanh | -0.1962 [-0.4165, +0.0241] | 8 | 1 | 1 | 0.80 [0.44, 0.97] |
| AIC | +0.0082 [-0.1289, +0.1454] | 4 | 5 | 1 | 0.40 [0.12, 0.74] |
| true family | -0.1870 [-0.3662, -0.0078] | 7 | 2 | 1 | 0.70 [0.35, 0.93] |

Of 30 channels, AIC picked the world's family (weibull) in 8, and the keep rule kept it in 22; families kept a channel: mean 2.87, distinct readings 2.03; the robust plan's readings, median 8, most 18. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 3, weibull 8, logistic 26; fits with a negative coefficient, planned as nought: tanh 0, exponential 0, michaelis-menten 0, hill 0, weibull 0, logistic 0.

## logistic-4: 10 worlds from seed 900

| arm | mean [95 %] | median | failed |
|---|---|---|---|
| status quo | 0.7513 [0.5167, 0.9859] | 0.7288 | 0 |
| tanh | 0.6941 [0.4516, 0.9365] | 0.7003 | 0 |
| AIC | 0.3731 [0.0307, 0.7154] | 0.1280 | 0 |
| robust | 0.3564 [0.0063, 0.7064] | 0.0445 | 0 |
| true family | 0.7513 [0.5167, 0.9859] | 0.7288 | 10 |

The robust plan against each arm: its regret less the arm's over the same worlds, and the worlds where it was lower, tied or higher (Clopper-Pearson 95 % on the lower share).

| against | difference [95 %] | lower | tied | higher | lower share [95 %] |
|---|---|---|---|---|---|
| status quo | -0.3950 [-0.6841, -0.1058] | 9 | 0 | 1 | 0.90 [0.55, 1.00] |
| tanh | -0.3377 [-0.6026, -0.0728] | 6 | 4 | 0 | 0.60 [0.26, 0.88] |
| AIC | -0.0167 [-0.0545, +0.0211] | 1 | 9 | 0 | 0.10 [0.00, 0.45] |
| true family | -0.3950 [-0.6841, -0.1058] | 9 | 0 | 1 | 0.90 [0.55, 1.00] |

Of 30 channels, AIC picked the world's family (logistic) in 3, and the keep rule kept it in 4; families kept a channel: mean 2.43, distinct readings 1.60; the robust plan's readings, median 4, most 8. Fits that raised, by family: tanh 0, exponential 0, michaelis-menten 0, hill 8, weibull 11, logistic 26; fits with a negative coefficient, planned as nought: tanh 1, exponential 0, michaelis-menten 1, hill 0, weibull 0, logistic 0.
