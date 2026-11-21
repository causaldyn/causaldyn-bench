# The dither contrast under a persistent covariate

Track Q v3's caveat (`results/track_q_v3.md`): on the true graph, `prescribe(dither=)`'s contrast rejected 0.155 of 200 panels at 0.05 in confounder AR and 0.000 in every other world. `causaldyn_bench.dither_persistence`, seed 20261121, panels of 400 units x 12 periods, the logger's draw logged as `fresh`, `JAX_ENABLE_X64` True. Runs: **base** on causal-hybrid-control 1.3.1 at `6ceb884`; **fixed** on causal-hybrid-control 1.3.1 at `0cc1407`. Each run draws the same panels.

## Size: the true graph, the promotion's persistence set

Confounder AR's world (a promotion the logger reads, unit variance, pushing supply by 0.2 a period) at each persistence; the oracle arm's graph. The lean is the slope of the promotion's push on the state that a drift on the states alone reads, per unit of time, over the panel's 11 transitions (`scripts/dither_persistence_lean.mac`).

| setting | world | persistence | lean on (supply, wait) | base: rejected at 0.05 (± MC error) [95%] | fixed: rejected at 0.05 (± MC error) [95%] |
|---|---|---|---|---|---|
| size 0 | confounder AR | 0 | (0.000, 0.000) | 0.000 ± 0.000 [0.000, 0.018] | 0.035 ± 0.013 [0.014, 0.071] |
| size 0.5 | confounder AR | 0.5 | (0.587, 0.159) | 0.000 ± 0.000 [0.000, 0.018] | 0.010 ± 0.007 [0.001, 0.036] |
| size 0.8 | confounder AR | 0.8 | (0.956, 0.261) | 0.085 ± 0.020 [0.050, 0.133] | 0.040 ± 0.014 [0.017, 0.077] |
| size 0.95 | confounder AR | 0.95 | (1.153, 0.192) | 0.265 ± 0.031 [0.205, 0.332] | 0.030 ± 0.012 [0.011, 0.064] |

## Power: the wrong graph, where Track Q v3's gate passed

| setting | world | the wrong graph | base: rejected at 0.05 (± MC error) [95%] | fixed: rejected at 0.05 (± MC error) [95%] |
|---|---|---|---|---|
| power mediator | mediator | turns incentive -> orders around, so the fit adjusts for a confounded mediator | 1.000 ± 0.000 [0.929, 1.000] | 1.000 ± 0.000 [0.929, 1.000] |
| power collider latent | collider latent | makes sessions, moved by the incentive and a latent that moves supply, a parent | 1.000 ± 0.000 [0.929, 1.000] | 1.000 ± 0.000 [0.929, 1.000] |
| power confounder AR | confounder AR | omits the same promotion, persistent | 0.800 ± 0.057 [0.663, 0.900] | 0.920 ± 0.038 [0.808, 0.978] |

## The p-values' law, every cell

The rates at 0.1 and 0.2, the median p-value, and on average the priced cost `(d' W d - tr(W S)) / 2` and the noise `tr(W S) / 2`, the variance the contrast reads: on the true graph the cost averages 0 where `S` is the difference's variance and the two fits estimate one thing.

| setting | run | rejected at 0.1 | at 0.2 | median p | mean cost | mean noise |
|---|---|---|---|---|---|---|
| size 0 | base | 0.000 | 0.000 | 0.800 | -4.14e-05 | 4.86e-05 |
| size 0 | fixed | 0.060 | 0.130 | 0.544 | -2.71e-06 | 1.09e-05 |
| size 0.5 | base | 0.030 | 0.095 | 0.462 | -9.38e-06 | 4.66e-05 |
| size 0.5 | fixed | 0.090 | 0.180 | 0.546 | -2.42e-06 | 1.22e-05 |
| size 0.8 | base | 0.145 | 0.355 | 0.252 | 3.78e-05 | 5.25e-05 |
| size 0.8 | fixed | 0.090 | 0.235 | 0.571 | -1.1e-06 | 1.58e-05 |
| size 0.95 | base | 0.380 | 0.660 | 0.142 | 0.000112 | 5.15e-05 |
| size 0.95 | fixed | 0.090 | 0.155 | 0.573 | -5.01e-06 | 1.88e-05 |
| power mediator | base | 1.000 | 1.000 | 0.000 | 0.0929 | 6.4e-05 |
| power mediator | fixed | 1.000 | 1.000 | 0.000 | 0.0953 | 5.87e-05 |
| power collider latent | base | 1.000 | 1.000 | 0.000 | 0.0722 | 5.18e-05 |
| power collider latent | fixed | 1.000 | 1.000 | 0.000 | 0.0736 | 5.2e-05 |
| power confounder AR | base | 0.860 | 0.920 | 0.000 | 0.000862 | 5.02e-05 |
| power confounder AR | fixed | 0.940 | 0.960 | 0.000 | 0.000888 | 2.55e-05 |

Void panels: base 0; fixed 0. The MC error is the binomial standard error at the measured rate; the interval is Clopper-Pearson's. Wall-clock is recorded in the JSON and quoted nowhere: it was not measured on a clean machine.
