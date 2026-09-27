# P3.3 after the gate -- POST HOC, not pre-registered

Produced by `causaldyn_bench.boptest_capped_posthoc` from the scored run's
`results.json` (its verdict: **INCONCLUSIVE**, precision float64); nothing here changes that verdict.

- precision of this run: **float64** (`JAX_ENABLE_X64`)
- `theta_ref = 0.6387` K/h; the stage-0 PRBS fit's authority at the target `0.5175`, i.e. `0.810` theta_ref

## Same energy, same estimate?

Final estimate, taper minus block, over 168 pairs: mean **-0.0266** K/h, 95% interval [-0.0433, -0.0099] (stratified bootstrap); the taper ended lower in 92 of 168. Mean final estimates: taper `0.6159`, block `0.6425`, against `theta_ref = 0.6387`.

| prior | pairs | taper regret | block regret | mean D | taper wins | taper estimate | block estimate | lazy regret |
|---|---|---|---|---|---|---|---|---|
| above theta_ref | 84 | 7.2872 | 9.2130 | -1.9258 | 53/84 | 0.6791 | 0.7122 | 11.3358 |
| below theta_ref | 84 | 11.5342 | 10.2251 | 1.3091 | 33/84 | 0.5526 | 0.5727 | 21.4155 |

## Per window

| window | day | mean D | taper wins | taper estimate | block estimate | lowest m |
|---|---|---|---|---|---|---|
| 0 | 1 | -3.7084 | 19/24 | 0.5925 | 0.6654 | 0.8 |
| 1 | 9 | 0.4209 | 11/24 | 0.6605 | 0.6215 | 1 |
| 2 | 17 | -1.7584 | 14/24 | 0.6330 | 0.7213 | 1 |
| 3 | 25 | -1.5441 | 12/24 | 0.5501 | 0.5537 | 1.1 |
| 4 | 33 | 2.8915 | 5/24 | 0.6225 | 0.6995 | 1.1 |
| 5 | 41 | -1.0717 | 13/24 | 0.6030 | 0.4980 | 1.1 |
| 6 | 49 | 2.6117 | 12/24 | 0.6494 | 0.7379 | 1.25 |

## Is exploring priced at one constant A?

Learning switched off: the estimate held at `m theta_ref` (it moved at most `3.2e-11` K/h), the scored budget spent by each schedule on the scored signs;
cost = loss above the same controller's scored no-probe episode, per unit of probe
energy, against the plan's `A`.

| m | episodes | block energy | taper energy | block cost / energy | taper cost / energy | block minus taper | block dearer | model A |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 14 | 1.3613 | 1.3044 | 3.3305 | 3.2458 | 0.0847 | 8/14 | 1.6320 |
| 1 | 14 | 1.3613 | 1.3613 | 1.9500 | 2.5748 | -0.6247 | 7/14 | 1.6320 |
| 1.5 | 14 | 1.3613 | 1.3613 | 3.3512 | 3.5090 | -0.1577 | 8/14 | 1.6320 |

## Where is the regret lowest?

Regret of the no-probe controller acting on a fixed `m theta_ref`, against the scored
oracle (`m = 1`), per window [K^2 per episode]; the `m = 1` and lazy-arm episodes
reproduced the scored ones exactly.

| m | window 0 | window 1 | window 2 | window 3 | window 4 | window 5 | window 6 | mean |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 6.2442 | 12.3507 | 10.4109 | 26.5117 | 25.2551 | 33.7253 | 35.4110 | 21.4155 |
| 0.6 | 1.7706 | 8.0147 | 7.3236 | 19.6374 | 19.9732 | 24.8054 | 25.9191 | 15.3491 |
| 0.7 | -0.9379 | 4.4550 | 3.8639 | 12.5872 | 13.1331 | 15.5960 | 15.7949 | 9.2132 |
| 0.8 | -2.4003 | 1.7824 | 1.4777 | 6.6227 | 7.4632 | 6.3585 | 6.8423 | 4.0209 |
| 0.9 | -1.5239 | 0.4692 | 0.2103 | 2.5202 | 3.1060 | 4.4978 | 4.8666 | 2.0209 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1.1 | 3.2209 | 1.2828 | 1.5050 | -0.7115 | -1.4676 | -3.3432 | -3.9025 | -0.4880 |
| 1.25 | 10.3584 | 4.9271 | 5.7094 | 1.2151 | -0.8016 | -2.5042 | -4.5878 | 2.0452 |
| 1.5 | 27.6224 | 16.4672 | 18.9269 | 12.5433 | 6.6341 | -0.2809 | -2.5627 | 11.3358 |

- lowest mean regret at `m = 1.1`; lowest per window at `m = 0.8, 1, 1, 1.1, 1.1, 1.1, 1.25`
