# L8.1 -- BOPTEST through `chc.prescribe`

Produced by `causaldyn_bench.boptest_prescribe` at its pre-registered defaults; the
pre-registration is that module's docstring, committed before this run.

- test case `bestest_hydronic_heat_pump`, steps of 30 min; each of 6 replicates logs 960 steps of the reset policy, then runs 336-step windows
- target offsets -0.5, -0.25, +0, +0.25, +0.5, +1, +1.5 K; horizon 16 steps, tolerance 0.5 K, lever unit cost 0.0
- precision: **float64** (`JAX_ENABLE_X64`); chc 0.5.1

## The gate

`excess = E_naive / E_adjusted - 1`, each arm's `ener_tot` read off its own front at the built-in baseline's `tdis_tot` in the same window (relative, per replicate): mean **-0.1165**, 95% t-interval [-0.2276, -0.0054] over 4 of 6 replicates; the adjusted arm was cheaper in 0.

- decision: **REFUTED**
- V1 (both fronts reach the baseline's comfort in every replicate): FAIL
- V2 (no episode had more than 0.01 of its solves stopped by the iteration budget): worst 0.152 -> FAIL
- the verdict DOES NOT STAND

## Against the built-in baseline (descriptive, no decision)

`saving = 1 - E_arm / E_baseline`, the arm's energy read at the baseline's discomfort.

| arm | replicates | mean saving | 95% t-interval |
|---|---|---|---|
| adjusted | 5 | +0.1045 | [+0.0857, +0.1233] |
| naive | 5 | +0.2112 | [+0.1352, +0.2872] |

## Per replicate

Energy in kWh/m2 and discomfort in K h, BOPTEST's `ener_tot` and `tdis_tot`. `--` where a front does not reach the level.

| replicate | log from day | window from day | baseline tdis | baseline ener | adjusted ener at baseline tdis | naive ener at baseline tdis | excess | adjusted tdis at baseline ener | naive tdis at baseline ener |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 5.1359 | 1.7260 | 1.5813 | 1.4436 | -0.0871 | -- | -- |
| 1 | 7 | 27 | 3.4514 | 1.8867 | -- | 1.4391 | -- | -- | -- |
| 2 | 14 | 34 | 5.9027 | 1.7812 | 1.5764 | 1.4143 | -0.1028 | -- | -- |
| 3 | 21 | 41 | 5.6058 | 1.6424 | 1.4856 | 1.3987 | -0.0585 | -- | -- |
| 4 | 28 | 48 | 4.2318 | 1.6473 | 1.4717 | 1.1515 | -0.2176 | -- | -- |
| 5 | 35 | 55 | 3.7969 | 1.4327 | 1.2583 | -- | -- | -- | -- |

## What `prescribe` identified from each log

Read through Track D-causal's definitions: the decay at the log's mean action and at the two ends of the actuator, the authority at 21 C in K/h per unit modulation, the 8-hour step response in K.

| replicate | arm | identification | adjusted for | authority at 21 C | 8 h step response | decay at mean action | decay off / full | channel standard error | overlap |
|---|---|---|---|---|---|---|---|---|---|
| 0 | adjusted | identified | outdoor, solar, bound | 0.7104 | 5.322 | -0.01766 | -0.00993 / -0.03447 | 0.5085 | 0.01132 |
| 0 | naive | asserted | nothing | 0.4093 | 3.073 | -0.01709 | -0.00021 / -0.05382 | -- | 0.07373 |
| 1 | adjusted | identified | outdoor, solar, bound | 0.4892 | 3.688 | -0.01596 | +0.01729 / -0.10269 | 0.6108 | 0.00955 |
| 1 | naive | asserted | nothing | 0.3107 | 2.353 | -0.01469 | +0.01337 / -0.08789 | -- | 0.06415 |
| 2 | adjusted | identified | outdoor, solar, bound | 0.6031 | 4.521 | -0.01749 | -0.00545 / -0.04870 | 0.6408 | 0.00936 |
| 2 | naive | asserted | nothing | 0.2736 | 2.048 | -0.01795 | +0.02136 / -0.11994 | -- | 0.06710 |
| 3 | adjusted | identified | outdoor, solar, bound | 0.6261 | 4.474 | -0.03056 | -0.02492 / -0.04328 | 0.5615 | 0.01109 |
| 3 | naive | asserted | nothing | 0.3616 | 2.637 | -0.02497 | -0.00167 / -0.07746 | -- | 0.06713 |
| 4 | adjusted | identified | outdoor, solar, bound | 0.7785 | 6.296 | +0.00291 | -0.01979 / +0.04898 | 0.5851 | 0.01133 |
| 4 | naive | asserted | nothing | 0.2645 | 1.970 | -0.01921 | +0.02355 / -0.10602 | -- | 0.07317 |
| 5 | adjusted | identified | outdoor, solar, bound | 0.5720 | 4.212 | -0.02237 | +0.00201 / -0.07707 | 0.6050 | 0.01002 |
| 5 | naive | asserted | nothing | 0.1820 | 1.332 | -0.02393 | +0.01899 / -0.12025 | -- | 0.06609 |

## Fronts

| replicate | arm | offset K | tdis K h | ener kWh/m2 | cost | saturated | mean action |
|---|---|---|---|---|---|---|---|
| 0 | adjusted | -0.5 | 49.1336 | 1.5066 | 0.3819 | 0.348 | 0.301 |
| 0 | adjusted | -0.25 | 25.0790 | 1.5402 | 0.3904 | 0.348 | 0.311 |
| 0 | adjusted | +0 | 3.5621 | 1.5846 | 0.4017 | 0.330 | 0.320 |
| 0 | adjusted | +0.25 | 0.0000 | 1.6046 | 0.4068 | 0.336 | 0.332 |
| 0 | adjusted | +0.5 | 0.0000 | 1.6251 | 0.4120 | 0.342 | 0.344 |
| 0 | adjusted | +1 | 0.0000 | 1.6861 | 0.4274 | 0.348 | 0.366 |
| 0 | adjusted | +1.5 | 0.0000 | 1.7552 | 0.4449 | 0.339 | 0.388 |
| 0 | naive | -0.5 | 48.7345 | 1.3919 | 0.3528 | 0.491 | 0.318 |
| 0 | naive | -0.25 | 24.7003 | 1.4174 | 0.3593 | 0.491 | 0.330 |
| 0 | naive | +0 | 3.6462 | 1.4673 | 0.3720 | 0.506 | 0.340 |
| 0 | naive | +0.25 | 0.0052 | 1.4504 | 0.3677 | 0.551 | 0.357 |
| 0 | naive | +0.5 | 0.0000 | 1.4913 | 0.3781 | 0.557 | 0.368 |
| 0 | naive | +1 | 0.0000 | 1.5490 | 0.3927 | 0.592 | 0.392 |
| 0 | naive | +1.5 | 0.0000 | 1.5963 | 0.4047 | 0.619 | 0.418 |
| 1 | adjusted | -0.5 | 51.8252 | 1.5891 | 0.4028 | 0.348 | 0.317 |
| 1 | adjusted | -0.25 | 26.2830 | 1.5961 | 0.4046 | 0.363 | 0.331 |
| 1 | adjusted | +0 | 3.3254 | 1.5918 | 0.4035 | 0.393 | 0.345 |
| 1 | adjusted | +0.25 | 0.0000 | 1.5655 | 0.3969 | 0.449 | 0.363 |
| 1 | adjusted | +0.5 | 0.0000 | 1.5845 | 0.4017 | 0.470 | 0.379 |
| 1 | adjusted | +1 | 0.0000 | 1.6477 | 0.4177 | 0.542 | 0.406 |
| 1 | adjusted | +1.5 | 0.0000 | 1.6719 | 0.4238 | 0.655 | 0.439 |
| 1 | naive | -0.5 | 50.6666 | 1.4003 | 0.3550 | 0.557 | 0.341 |
| 1 | naive | -0.25 | 24.6353 | 1.4112 | 0.3577 | 0.607 | 0.359 |
| 1 | naive | +0 | 3.7352 | 1.4390 | 0.3648 | 0.649 | 0.370 |
| 1 | naive | +0.25 | 0.0000 | 1.4402 | 0.3651 | 0.708 | 0.391 |
| 1 | naive | +0.5 | 0.0000 | 1.4631 | 0.3709 | 0.753 | 0.404 |
| 1 | naive | +1 | 0.0000 | 1.5296 | 0.3878 | 0.792 | 0.424 |
| 1 | naive | +1.5 | 0.0000 | 1.5754 | 0.3994 | 0.857 | 0.455 |
| 2 | adjusted | -0.5 | 51.2791 | 1.5338 | 0.3888 | 0.378 | 0.326 |
| 2 | adjusted | -0.25 | 26.8843 | 1.5681 | 0.3975 | 0.372 | 0.336 |
| 2 | adjusted | +0 | 4.8547 | 1.5769 | 0.3997 | 0.390 | 0.349 |
| 2 | adjusted | +0.25 | 0.0505 | 1.6218 | 0.4111 | 0.378 | 0.358 |
| 2 | adjusted | +0.5 | 0.0000 | 1.6464 | 0.4174 | 0.390 | 0.369 |
| 2 | adjusted | +1 | 0.0000 | 1.7083 | 0.4331 | 0.390 | 0.392 |
| 2 | adjusted | +1.5 | 0.0000 | 1.7644 | 0.4473 | 0.405 | 0.415 |
| 2 | naive | -0.5 | 49.7761 | 1.4086 | 0.3571 | 0.557 | 0.346 |
| 2 | naive | -0.25 | 25.3893 | 1.4229 | 0.3607 | 0.628 | 0.361 |
| 2 | naive | +0 | 4.7448 | 1.4145 | 0.3586 | 0.690 | 0.380 |
| 2 | naive | +0.25 | 0.0157 | 1.4475 | 0.3669 | 0.726 | 0.391 |
| 2 | naive | +0.5 | 0.0000 | 1.4580 | 0.3696 | 0.759 | 0.404 |
| 2 | naive | +1 | 0.0000 | 1.4922 | 0.3783 | 0.878 | 0.433 |
| 2 | naive | +1.5 | 966.2751 | 3.4986 | 0.8869 | 0.997 | 0.952 |
| 3 | adjusted | -0.5 | 51.4591 | 1.4087 | 0.3571 | 0.443 | 0.332 |
| 3 | adjusted | -0.25 | 28.0456 | 1.4456 | 0.3665 | 0.446 | 0.345 |
| 3 | adjusted | +0 | 6.2107 | 1.4826 | 0.3758 | 0.455 | 0.357 |
| 3 | adjusted | +0.25 | 0.0370 | 1.5135 | 0.3837 | 0.467 | 0.371 |
| 3 | adjusted | +0.5 | 0.0000 | 1.5450 | 0.3917 | 0.473 | 0.384 |
| 3 | adjusted | +1 | 0.0000 | 1.6256 | 0.4121 | 0.500 | 0.409 |
| 3 | adjusted | +1.5 | 0.0000 | 1.6940 | 0.4294 | 0.503 | 0.435 |
| 3 | naive | -0.5 | 50.2606 | 1.3580 | 0.3443 | 0.580 | 0.345 |
| 3 | naive | -0.25 | 26.3964 | 1.3586 | 0.3444 | 0.631 | 0.363 |
| 3 | naive | +0 | 5.4034 | 1.3991 | 0.3547 | 0.634 | 0.375 |
| 3 | naive | +0.25 | 0.0114 | 1.4388 | 0.3647 | 0.637 | 0.389 |
| 3 | naive | +0.5 | 0.0000 | 1.4779 | 0.3746 | 0.658 | 0.402 |
| 3 | naive | +1 | 0.0000 | 1.5178 | 0.3848 | 0.741 | 0.432 |
| 3 | naive | +1.5 | 0.0000 | 1.5702 | 0.3980 | 0.798 | 0.457 |
| 4 | adjusted | -0.5 | 48.5736 | 1.4003 | 0.3550 | 0.360 | 0.236 |
| 4 | adjusted | -0.25 | 23.5429 | 1.4275 | 0.3619 | 0.351 | 0.245 |
| 4 | adjusted | +0 | 1.9260 | 1.4770 | 0.3744 | 0.327 | 0.253 |
| 4 | adjusted | +0.25 | 0.0000 | 1.5194 | 0.3852 | 0.312 | 0.261 |
| 4 | adjusted | +0.5 | 0.0000 | 1.5380 | 0.3899 | 0.315 | 0.272 |
| 4 | adjusted | +1 | 0.0000 | 1.5898 | 0.4030 | 0.324 | 0.292 |
| 4 | adjusted | +1.5 | 0.0000 | 1.6435 | 0.4166 | 0.324 | 0.313 |
| 4 | naive | -0.5 | 47.4514 | 1.1100 | 0.2814 | 0.658 | 0.271 |
| 4 | naive | -0.25 | 22.0755 | 1.1545 | 0.2927 | 0.658 | 0.280 |
| 4 | naive | +0 | 2.7907 | 1.1529 | 0.2923 | 0.717 | 0.296 |
| 4 | naive | +0.25 | 0.0000 | 1.1703 | 0.2967 | 0.762 | 0.308 |
| 4 | naive | +0.5 | 0.0000 | 1.1832 | 0.2999 | 0.821 | 0.324 |
| 4 | naive | +1 | 0.0000 | 1.2469 | 0.3161 | 0.872 | 0.349 |
| 4 | naive | +1.5 | 718.3955 | 2.8269 | 0.7166 | 0.979 | 0.763 |
| 5 | adjusted | -0.5 | 42.0902 | 1.1900 | 0.3017 | 0.464 | 0.219 |
| 5 | adjusted | -0.25 | 20.8499 | 1.2321 | 0.3123 | 0.455 | 0.228 |
| 5 | adjusted | +0 | 2.3437 | 1.2605 | 0.3195 | 0.452 | 0.237 |
| 5 | adjusted | +0.25 | 0.0228 | 1.2782 | 0.3240 | 0.455 | 0.248 |
| 5 | adjusted | +0.5 | 0.4098 | 1.2795 | 0.3243 | 0.482 | 0.261 |
| 5 | adjusted | +1 | 1.9429 | 1.3164 | 0.3337 | 0.506 | 0.287 |
| 5 | adjusted | +1.5 | 4.5195 | 1.3485 | 0.3418 | 0.551 | 0.314 |
| 5 | naive | -0.5 | 60.6279 | 1.1658 | 0.2955 | 0.804 | 0.314 |
| 5 | naive | -0.25 | 524.0294 | 2.3529 | 0.5964 | 0.896 | 0.629 |
| 5 | naive | +0 | 522.3347 | 2.3542 | 0.5968 | 0.943 | 0.639 |
| 5 | naive | +0.25 | 1364.8750 | 3.6962 | 0.9370 | 0.991 | 0.951 |
| 5 | naive | +0.5 | 1370.2950 | 3.7032 | 0.9388 | 0.994 | 0.954 |
| 5 | naive | +1 | 1403.0512 | 3.7512 | 0.9509 | 1.000 | 0.967 |
| 5 | naive | +1.5 | 1402.6044 | 3.7513 | 0.9509 | 0.997 | 0.967 |

## What the certificate said, and what the plant did

Every control step is one `prescribe` call, and only its first action is applied, so the one step ahead is the part of each plan the plant can check. Pooled over every episode of an arm: the trustworthy prefix each call reported, the error tube's radius one step ahead, the plant's one-step error against the plan's own next temperature, the share of steps that error stayed inside the tube and inside the tolerance, the largest finite regret bound and the share of calls whose bound was infinite (the objective not convex over the box), and the largest share of calls any episode had stopped by the iteration budget or left at the zero guess. `--` where the certificate did not evaluate the quantity. The regret bound is a gap in the planning objective on the fitted model; the certificate carries no barrier, because the façade takes no constraint on its target.

| arm | trustworthy steps | tube one step ahead K | one-step error rms K | largest error K | inside the tube | inside the tolerance | regret bound | bound infinite | stopped by budget | no progress |
|---|---|---|---|---|---|---|---|---|---|---|
| adjusted | 1-1 | 0.2926 | 0.1305 | 0.7931 | 0.959 | 0.991 | 2.15e-06 | 0.034 | 0.000 | 0.193 |
| naive | 0-0 | -- | 0.3263 | 1.6513 | -- | 0.872 | 5.34e-05 | 0.121 | 0.152 | 0.247 |

## `report()` for replicate 0

One call per arm from the log's last state towards 21 C: what the façade prints.

### adjusted

```text
# Prescription for `zone`

## Decision

| lever | active steps | first | last |
|---|---|---|---|
| `modulation` | 0-15 | +1 | +0.355 |

Planned task cost: 0.17414.

## Certificate

- identification: **identified** (blocks every non-causal path from ['modulation'] to ['zone'])
- adjusted for: ['outdoor', 'solar', 'bound']
- channel standard error: 0.5085
- overlap (residualised action variance): 0.01132
- error tube: **partial**, certified horizon 1
- barrier: certified steps not evaluated, gamma* not evaluated
- solver: converged after 465 accepted steps

**Trustworthy prefix: 1 steps.**

## Provenance

- data sha256: `403d845ea0f9bc19...`
- chc 0.5.1, 960 rows, x64=True, seed=0
```

### naive

```text
# Prescription for `zone`

## Decision

| lever | active steps | first | last |
|---|---|---|---|
| `modulation` | 0-15 | +1 | +0.382 |

Planned task cost: 0.241098.

## Certificate

- identification: **asserted** (asserted by the caller, not derived from a graph; nothing here checked it; the estimator was handed no covariate and no instrument, so its channel is the observational fit, which is the interventional one only if the lever is unconfounded)
- adjusted for: nothing
- channel standard error: not evaluated
- overlap (residualised action variance): 0.07373
- error tube: **not_evaluated**, certified horizon not evaluated
- barrier: certified steps not evaluated, gamma* not evaluated
- solver: converged after 1350 accepted steps

**Trustworthy prefix: 0 steps.**

## Provenance

- data sha256: `403d845ea0f9bc19...`
- chc 0.5.1, 960 rows, x64=True, seed=0
```
