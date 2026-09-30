# D21 -- L8.1 rerun: BOPTEST through `chc.prescribe`, the band held, the weather as drivers, the bound as a schedule

Produced by `causaldyn_bench.boptest_rerun` at its pre-registered defaults; the
pre-registration is that module's docstring, committed before this run.

- test case `bestest_hydronic_heat_pump`, steps of 30 min; each of 6 replicates logs 960 steps of the reset policy, then runs 336-step windows
- target offsets +1, +1.25, +1.5, +1.75, +2, +2.25, +2.5 K over the lower bound; horizon 16 steps, tolerance 0.5 K, lever unit cost 0.0
- precision: **float64** (`JAX_ENABLE_X64`); chc 0.8.0 at `f3f9eaece23c`

## The gate

`excess = E_naive / E_adjusted - 1`, each arm's `ener_tot` read off its own front at the built-in baseline's `tdis_tot` in the same window: mean **+0.0088**, 95% t-interval [-0.0140, +0.0317] over 4 of 6 replicates; the adjusted arm was cheaper in 2.

- decision: **INCONCLUSIVE**
- V1 (both fronts reach the baseline's comfort in every replicate): FAIL
- V2 (no episode had more than 0.01 of its calls from inside the band stopped by the iteration budget): worst 0.071 -> FAIL
- the verdict DOES NOT STAND

## Against the built-in baseline (descriptive, no decision)

`saving = 1 - E_arm / E_baseline`, the arm's energy read at the baseline's discomfort.

| arm | replicates | mean saving | 95% t-interval |
|---|---|---|---|
| adjusted | 4 | +0.1473 | [+0.0681, +0.2265] |
| naive | 4 | +0.1402 | [+0.0758, +0.2046] |

## Per replicate

Energy in kWh/m2 and discomfort in K h, BOPTEST's `ener_tot` and `tdis_tot`. `--` where a front does not reach the level.

| replicate | log from day | window from day | baseline tdis | baseline ener | adjusted ener at baseline tdis | naive ener at baseline tdis | excess | adjusted tdis at baseline ener | naive tdis at baseline ener |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 5.1359 | 1.7260 | 1.4627 | 1.4627 | -0.0000 | -- | -- |
| 1 | 7 | 27 | 3.4514 | 1.8867 | -- | -- | -- | -- | -- |
| 2 | 14 | 34 | 5.9027 | 1.7812 | 1.5339 | 1.5429 | +0.0059 | -- | -- |
| 3 | 21 | 41 | 5.6058 | 1.6424 | 1.4973 | 1.4967 | -0.0004 | -- | -- |
| 4 | 28 | 48 | 4.2318 | 1.6473 | 1.3024 | 1.3414 | +0.0299 | -- | -- |
| 5 | 35 | 55 | 3.7969 | 1.4327 | -- | -- | -- | -- | -- |

## What `prescribe` identified from each log

The channel `b0 + b1 T` in K/h per unit modulation and the zone temperature where it changes sign, the drift `a0 + a1 T`, and the drivers' gains in K/h per C of outdoor air and per kW/m2 of sun.

| replicate | arm | identification | adjusted for | channel | channel zero at C | authority at 21 C | drift | outdoor gain | solar gain | channel standard error | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | adjusted | identified | outdoor, solar, bound | -0.396 +0.0602 T | 6.59 | 0.8674 | +0.534 -0.0472 T | 0.0257 | 2.2691 | 0.7551 | 0.01131 |
| 0 | naive | asserted | nothing | +0.492 +0.0164 T | -29.93 | 0.8365 | +0.311 -0.0366 T | 0.0270 | 2.3601 | -- | 0.05181 |
| 1 | adjusted | identified | outdoor, solar, bound | +3.579 -0.1505 T | 23.78 | 0.4187 | -0.417 +0.0049 T | 0.0209 | 2.0820 | 0.7627 | 0.00979 |
| 1 | naive | asserted | nothing | +2.795 -0.1046 T | 26.73 | 0.5989 | -0.221 -0.0078 T | 0.0251 | 2.2233 | -- | 0.03723 |
| 2 | adjusted | identified | outdoor, solar, bound | +1.954 -0.0657 T | 29.72 | 0.5735 | -0.109 -0.0110 T | 0.0210 | 1.9387 | 0.5807 | 0.00939 |
| 2 | naive | asserted | nothing | +1.709 -0.0471 T | 36.29 | 0.7202 | -0.045 -0.0172 T | 0.0255 | 2.1501 | -- | 0.04039 |
| 3 | adjusted | identified | outdoor, solar, bound | +0.743 -0.0035 T | 210.85 | 0.6686 | +0.333 -0.0340 T | 0.0207 | 1.9188 | 0.4711 | 0.01157 |
| 3 | naive | asserted | nothing | +1.322 -0.0279 T | 47.44 | 0.7370 | +0.178 -0.0284 T | 0.0244 | 2.0670 | -- | 0.04131 |
| 4 | adjusted | identified | outdoor, solar, bound | +1.646 -0.0505 T | 32.60 | 0.5856 | +0.411 -0.0373 T | 0.0151 | 2.2137 | 0.5495 | 0.01118 |
| 4 | naive | asserted | nothing | +0.998 -0.0126 T | 79.01 | 0.7331 | +0.545 -0.0466 T | 0.0190 | 2.3845 | -- | 0.04846 |
| 5 | adjusted | identified | outdoor, solar, bound | +1.576 -0.0485 T | 32.53 | 0.5588 | +0.467 -0.0387 T | 0.0114 | 2.0159 | 0.5864 | 0.01052 |
| 5 | naive | asserted | nothing | +1.288 -0.0246 T | 52.31 | 0.7711 | +0.519 -0.0456 T | 0.0173 | 2.2598 | -- | 0.03890 |

## Fronts

| replicate | arm | offset K | tdis K h | ener kWh/m2 | cost | saturated | mean action | ended above band | calls from above band |
|---|---|---|---|---|---|---|---|---|---|
| 0 | adjusted | +1 | 22.6257 | 1.3444 | 0.3408 | 0.762 | 0.375 | 0.000 | 0.000 |
| 0 | adjusted | +1.25 | 16.0884 | 1.3790 | 0.3496 | 0.765 | 0.386 | 0.000 | 0.000 |
| 0 | adjusted | +1.5 | 10.6367 | 1.4172 | 0.3593 | 0.765 | 0.395 | 0.000 | 0.000 |
| 0 | adjusted | +1.75 | 6.3169 | 1.4517 | 0.3680 | 0.762 | 0.405 | 0.000 | 0.000 |
| 0 | adjusted | +2 | 3.2833 | 1.4799 | 0.3752 | 0.777 | 0.416 | 0.000 | 0.000 |
| 0 | adjusted | +2.25 | 1.3557 | 1.5112 | 0.3831 | 0.783 | 0.426 | 0.000 | 0.000 |
| 0 | adjusted | +2.5 | 0.4747 | 1.5471 | 0.3922 | 0.780 | 0.436 | 0.012 | 0.012 |
| 0 | naive | +1 | 23.6178 | 1.3450 | 0.3409 | 0.768 | 0.376 | 0.000 | 0.000 |
| 0 | naive | +1.25 | 16.9052 | 1.3727 | 0.3480 | 0.780 | 0.386 | 0.000 | 0.000 |
| 0 | naive | +1.5 | 11.4237 | 1.4111 | 0.3577 | 0.774 | 0.396 | 0.000 | 0.000 |
| 0 | naive | +1.75 | 7.0644 | 1.4422 | 0.3656 | 0.783 | 0.406 | 0.000 | 0.000 |
| 0 | naive | +2 | 3.7207 | 1.4777 | 0.3746 | 0.792 | 0.417 | 0.000 | 0.000 |
| 0 | naive | +2.25 | 1.4719 | 1.5134 | 0.3836 | 0.792 | 0.427 | 0.000 | 0.000 |
| 0 | naive | +2.5 | 0.4403 | 1.5420 | 0.3909 | 0.792 | 0.438 | 0.006 | 0.006 |
| 1 | adjusted | +1 | 49.3284 | 1.3075 | 0.3315 | 0.884 | 0.373 | 0.000 | 0.000 |
| 1 | adjusted | +1.25 | 40.6712 | 1.3253 | 0.3360 | 0.911 | 0.383 | 0.000 | 0.000 |
| 1 | adjusted | +1.5 | 30.9626 | 1.3622 | 0.3453 | 0.914 | 0.394 | 0.000 | 0.000 |
| 1 | adjusted | +1.75 | 23.0411 | 1.4015 | 0.3553 | 0.943 | 0.411 | 0.000 | 0.000 |
| 1 | adjusted | +2 | 15.7195 | 1.4343 | 0.3636 | 0.964 | 0.423 | 0.000 | 0.000 |
| 1 | adjusted | +2.25 | 113.9934 | 1.8431 | 0.4672 | 0.988 | 0.535 | 0.176 | 0.173 |
| 1 | adjusted | +2.5 | 194.1575 | 2.0299 | 0.5146 | 0.985 | 0.580 | 0.244 | 0.241 |
| 1 | naive | +1 | 43.9813 | 1.4112 | 0.3577 | 0.741 | 0.362 | 0.000 | 0.000 |
| 1 | naive | +1.25 | 35.4732 | 1.4288 | 0.3622 | 0.756 | 0.373 | 0.000 | 0.000 |
| 1 | naive | +1.5 | 26.9638 | 1.4390 | 0.3648 | 0.783 | 0.385 | 0.000 | 0.000 |
| 1 | naive | +1.75 | 20.0301 | 1.4406 | 0.3652 | 0.812 | 0.400 | 0.000 | 0.000 |
| 1 | naive | +2 | 13.5166 | 1.4908 | 0.3779 | 0.810 | 0.411 | 0.000 | 0.000 |
| 1 | naive | +2.25 | 8.3947 | 1.5195 | 0.3852 | 0.827 | 0.422 | 0.000 | 0.000 |
| 1 | naive | +2.5 | 5.1312 | 1.5505 | 0.3930 | 0.851 | 0.432 | 0.000 | 0.000 |
| 2 | adjusted | +1 | 37.0358 | 1.3758 | 0.3488 | 0.824 | 0.393 | 0.000 | 0.000 |
| 2 | adjusted | +1.25 | 29.2929 | 1.4017 | 0.3553 | 0.830 | 0.403 | 0.000 | 0.000 |
| 2 | adjusted | +1.5 | 20.9203 | 1.4371 | 0.3643 | 0.827 | 0.413 | 0.000 | 0.000 |
| 2 | adjusted | +1.75 | 14.1731 | 1.4723 | 0.3732 | 0.827 | 0.423 | 0.000 | 0.000 |
| 2 | adjusted | +2 | 9.1858 | 1.5048 | 0.3815 | 0.827 | 0.434 | 0.000 | 0.000 |
| 2 | adjusted | +2.25 | 5.5202 | 1.5373 | 0.3897 | 0.842 | 0.445 | 0.000 | 0.000 |
| 2 | adjusted | +2.5 | 3.2201 | 1.5706 | 0.3981 | 0.842 | 0.454 | 0.000 | 0.000 |
| 2 | naive | +1 | 34.7685 | 1.3835 | 0.3507 | 0.804 | 0.392 | 0.000 | 0.000 |
| 2 | naive | +1.25 | 26.8732 | 1.4145 | 0.3586 | 0.804 | 0.403 | 0.000 | 0.000 |
| 2 | naive | +1.5 | 19.6917 | 1.4422 | 0.3656 | 0.815 | 0.413 | 0.000 | 0.000 |
| 2 | naive | +1.75 | 14.7653 | 1.4775 | 0.3745 | 0.818 | 0.422 | 0.000 | 0.000 |
| 2 | naive | +2 | 9.6985 | 1.5111 | 0.3831 | 0.815 | 0.431 | 0.000 | 0.000 |
| 2 | naive | +2.25 | 5.4644 | 1.5466 | 0.3921 | 0.818 | 0.441 | 0.000 | 0.000 |
| 2 | naive | +2.5 | 2.7390 | 1.5818 | 0.4010 | 0.810 | 0.451 | 0.000 | 0.000 |
| 3 | adjusted | +1 | 33.9709 | 1.3322 | 0.3377 | 0.801 | 0.369 | 0.000 | 0.000 |
| 3 | adjusted | +1.25 | 26.1941 | 1.3621 | 0.3453 | 0.804 | 0.379 | 0.000 | 0.000 |
| 3 | adjusted | +1.5 | 18.7001 | 1.3891 | 0.3521 | 0.815 | 0.391 | 0.000 | 0.000 |
| 3 | adjusted | +1.75 | 13.1808 | 1.4287 | 0.3622 | 0.810 | 0.400 | 0.000 | 0.000 |
| 3 | adjusted | +2 | 8.0609 | 1.4712 | 0.3730 | 0.804 | 0.411 | 0.000 | 0.000 |
| 3 | adjusted | +2.25 | 5.0418 | 1.5033 | 0.3811 | 0.801 | 0.422 | 0.000 | 0.000 |
| 3 | adjusted | +2.5 | 3.1408 | 1.5377 | 0.3898 | 0.812 | 0.433 | 0.000 | 0.000 |
| 3 | naive | +1 | 31.4302 | 1.3359 | 0.3387 | 0.801 | 0.371 | 0.000 | 0.000 |
| 3 | naive | +1.25 | 23.3690 | 1.3697 | 0.3472 | 0.801 | 0.382 | 0.000 | 0.000 |
| 3 | naive | +1.5 | 16.9362 | 1.3927 | 0.3531 | 0.818 | 0.393 | 0.000 | 0.000 |
| 3 | naive | +1.75 | 11.3234 | 1.4363 | 0.3641 | 0.812 | 0.402 | 0.000 | 0.000 |
| 3 | naive | +2 | 7.5408 | 1.4728 | 0.3734 | 0.807 | 0.412 | 0.000 | 0.000 |
| 3 | naive | +2.25 | 4.1084 | 1.5152 | 0.3841 | 0.795 | 0.423 | 0.000 | 0.000 |
| 3 | naive | +2.5 | 2.4416 | 1.5433 | 0.3912 | 0.804 | 0.434 | 0.000 | 0.000 |
| 4 | adjusted | +1 | 19.5032 | 1.2137 | 0.3077 | 0.750 | 0.302 | 0.000 | 0.000 |
| 4 | adjusted | +1.25 | 12.5188 | 1.2440 | 0.3154 | 0.744 | 0.312 | 0.000 | 0.000 |
| 4 | adjusted | +1.5 | 7.6267 | 1.2712 | 0.3223 | 0.744 | 0.322 | 0.000 | 0.000 |
| 4 | adjusted | +1.75 | 3.5553 | 1.3087 | 0.3317 | 0.753 | 0.333 | 0.000 | 0.000 |
| 4 | adjusted | +2 | 1.8057 | 1.3391 | 0.3395 | 0.756 | 0.343 | 0.000 | 0.000 |
| 4 | adjusted | +2.25 | 0.5874 | 1.3791 | 0.3496 | 0.747 | 0.353 | 0.000 | 0.000 |
| 4 | adjusted | +2.5 | 0.2964 | 1.4112 | 0.3577 | 0.753 | 0.363 | 0.000 | 0.000 |
| 4 | naive | +1 | 16.7381 | 1.2320 | 0.3123 | 0.714 | 0.301 | 0.000 | 0.000 |
| 4 | naive | +1.25 | 11.3275 | 1.2664 | 0.3210 | 0.708 | 0.311 | 0.000 | 0.000 |
| 4 | naive | +1.5 | 7.3607 | 1.2999 | 0.3295 | 0.708 | 0.320 | 0.000 | 0.000 |
| 4 | naive | +1.75 | 3.7335 | 1.3480 | 0.3417 | 0.696 | 0.328 | 0.000 | 0.000 |
| 4 | naive | +2 | 1.3289 | 1.3759 | 0.3488 | 0.702 | 0.339 | 0.000 | 0.000 |
| 4 | naive | +2.25 | 0.4253 | 1.4078 | 0.3569 | 0.702 | 0.349 | 0.000 | 0.000 |
| 4 | naive | +2.5 | 0.1216 | 1.4366 | 0.3642 | 0.711 | 0.360 | 0.000 | 0.000 |
| 5 | adjusted | +1 | 20.3015 | 1.0387 | 0.2633 | 0.812 | 0.269 | 0.012 | 0.012 |
| 5 | adjusted | +1.25 | 14.8695 | 1.0666 | 0.2704 | 0.812 | 0.277 | 0.018 | 0.018 |
| 5 | adjusted | +1.5 | 10.4127 | 1.0990 | 0.2786 | 0.804 | 0.286 | 0.024 | 0.024 |
| 5 | adjusted | +1.75 | 7.4820 | 1.1341 | 0.2875 | 0.804 | 0.294 | 0.024 | 0.024 |
| 5 | adjusted | +2 | 5.7923 | 1.1603 | 0.2941 | 0.818 | 0.306 | 0.030 | 0.030 |
| 5 | adjusted | +2.25 | 4.8273 | 1.1933 | 0.3025 | 0.818 | 0.315 | 0.033 | 0.033 |
| 5 | adjusted | +2.5 | 4.9798 | 1.2160 | 0.3083 | 0.839 | 0.327 | 0.033 | 0.033 |
| 5 | naive | +1 | 18.5689 | 1.0479 | 0.2656 | 0.801 | 0.270 | 0.012 | 0.012 |
| 5 | naive | +1.25 | 13.0004 | 1.0779 | 0.2732 | 0.801 | 0.278 | 0.018 | 0.018 |
| 5 | naive | +1.5 | 9.0137 | 1.1103 | 0.2815 | 0.795 | 0.287 | 0.024 | 0.024 |
| 5 | naive | +1.75 | 6.1715 | 1.1467 | 0.2907 | 0.786 | 0.295 | 0.024 | 0.024 |
| 5 | naive | +2 | 4.5121 | 1.1727 | 0.2973 | 0.798 | 0.305 | 0.027 | 0.027 |
| 5 | naive | +2.25 | 4.4914 | 1.2059 | 0.3057 | 0.798 | 0.314 | 0.033 | 0.033 |
| 5 | naive | +2.5 | 4.7409 | 1.2390 | 0.3141 | 0.798 | 0.323 | 0.033 | 0.033 |

## What the certificate said, and what the plant did

Every control step is one `prescribe` call, and only its first action is applied, so the plant checks the plan's first step alone. Over every episode of each arm: the trustworthy prefix's range, the mean tube one step ahead, the one-step error's root mean square and largest size, the share of steps inside the tube and inside the tolerance, the largest finite regret bound, the share of calls whose bound was declined, the worst V2 share, and the calls from above the band that the budget stopped (not gated).

| arm | trusted | tube | error rms | error max | within tube | within tolerance | regret bound max | declined | budget stopped (V2) | stopped from above the band | barrier prefix min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adjusted | 0-2 | 0.3088 | 0.1072 | 0.7478 | 0.981 | 0.996 | 1.83e+01 | 0.072 | 0.071 | 192 | 0 |
| naive | 0-0 | -- | 0.0858 | 0.4106 | -- | 1.000 | 3.92e+01 | 0.021 | 0.055 | 54 | 0 |

## Against L8.1 (descriptive)

The same windows under L8.1's call and chc 0.5.1 (bench `46bb2f9`). Whether each replicate's panel is byte for byte L8.1's, and each baseline beside L8.1's.

| replicate | same panel | baseline tdis | L8.1 baseline tdis |
|---|---|---|---|
| 0 | True | 5.1359 | 5.1359 |
| 1 | True | 3.4514 | 3.4514 |
| 2 | True | 5.9027 | 5.9027 |
| 3 | True | 5.6058 | 5.6058 |
| 4 | True | 4.2318 | 4.2318 |
| 5 | True | 3.7969 | 3.7969 |

| arm | L8.1 mean saving | D21 mean saving |
|---|---|---|
| adjusted | +0.1045 | +0.1473 |
| naive | +0.2112 | +0.1402 |
