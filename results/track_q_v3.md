# Track Q, version 3 — the analyst's graph is wrong, and the stop reads the logger's draw

200 replicates a world (`causaldyn_bench.graph_errors_v3`, seed 20261026, x64 True), on causal-hybrid-control `a7bb967`, pre-registered in `results/track_q_v3_preregistration.md` at `ebbc69d`. On each replicate, one panel of 400 units x 12 periods to plan on, which logs the logger's draw as `fresh`, and a later one to evaluate on. Version 2's tolerance 0.8012 and threshold 0.05 + 0.0048 = 0.0548. The stop refuses a plan and its evaluation where `prescribe(dither=)`'s check was tested, rejects at 0.05, and prices the contrast above 0.1 of the gain the plan claims on its own model. Rate intervals are Clopper-Pearson's.

## The gate

The stop must remove failures more often than it refuses sound claims, as a stop at the same rate that ignored the failures would not (one-sided Fisher exact p), over the true and the wrong graph, or the wrong one alone where the two are the same:

| world | the wrong graph | failed, removed | sound, refused | p |
|---|---|---|---|---|
| none | is the true graph | 18, 0 | 182, 0 | 1 |
| chase | omits the logger's reading of demand | 24, 0 | 376, 0 | 1 |
| sticky | cannot state that the logger keeps half its last incentive | 16, 0 | 184, 0 | 1 |
| confounder | omits a promotion the logger reads and that moves supply | 159, 0 | 241, 0 | 1 |
| confounder AR | omits the same promotion, persistent | 5, 4 | 395, 0 | 4.8e-09 |
| mediator | turns incentive -> orders around, so the fit adjusts for a confounded mediator | 222, 200 | 178, 0 | 1.2e-89 |
| collider observed | makes sessions, moved by the incentive and demand, a parent of the incentive | 23, 0 | 377, 0 | 1 |
| collider latent | makes sessions, moved by the incentive and a latent that moves supply, a parent | 216, 200 | 184, 0 | 5.9e-96 |
| non-ancestor | says the incentive does not move supply | 44, 0 | 356, 0 | 1 |

**The stop lowers the silent-failure rate beyond chance: yes**, in confounder AR, mediator, collider latent. At 0.0056, 0.05 over the 9 worlds: confounder AR, mediator, collider latent.

**The contrast's size**, its rejection rate at 0.05 on the panels the true graph planned on, pooled over the worlds with an oracle: 0.019 over 1600 panels, 95% [0.013, 0.027], 0 of them void; at most 0.07: yes.

## The contrast by world

Per arm that read the draw: the panels tested and void, the contrast's rejection rate, the stop's rate, and over the tested panels the mean priced cost and the mean margin it is held to.

| world | arm | tested | void | rejected | stopped | cost | margin |
|---|---|---|---|---|---|---|---|
| none | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.542e-05 | 0.004698 |
| none | wrong | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.542e-05 | 0.004698 |
| chase | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.496e-05 | 0.004593 |
| chase | wrong | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.496e-05 | 0.004593 |
| sticky | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.485e-05 | 0.00472 |
| sticky | wrong | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.485e-05 | 0.00472 |
| confounder | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -4.056e-05 | 0.004691 |
| confounder | wrong | 200 | 0 | 0.995 [0.972, 1.000] | 0.000 | 0.001426 | 0.008377 |
| confounder AR | oracle | 200 | 0 | 0.155 [0.108, 0.213] | 0.010 | 6.078e-05 | 0.004711 |
| confounder AR | wrong | 200 | 0 | 0.790 [0.727, 0.844] | 0.010 | 0.0008045 | 0.008193 |
| mediator | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.37e-05 | 0.004636 |
| mediator | wrong | 200 | 0 | 1.000 [0.982, 1.000] | 1.000 | 0.09125 | 0.001011 |
| collider observed | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.397e-05 | 0.00467 |
| collider observed | wrong | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.391e-05 | 0.00467 |
| collider latent | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.414e-05 | 0.004659 |
| collider latent | wrong | 200 | 0 | 1.000 [0.982, 1.000] | 1.000 | 0.07435 | 1.642e-06 |
| non-ancestor | oracle | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.391e-05 | 0.00459 |
| non-ancestor | wrong | 200 | 0 | 0.000 [0.000, 0.018] | 0.000 | -1.391e-05 | 0.00459 |

## Secondary readings, not criteria

**The contrast alone as the stop**, at any priced cost:

| world | the wrong graph | failed, removed | sound, refused | p |
|---|---|---|---|---|
| none | is the true graph | 18, 0 | 182, 0 | 1 |
| chase | omits the logger's reading of demand | 24, 0 | 376, 0 | 1 |
| sticky | cannot state that the logger keeps half its last incentive | 16, 0 | 184, 0 | 1 |
| confounder | omits a promotion the logger reads and that moves supply | 159, 158 | 241, 41 | 5.8e-71 |
| confounder AR | omits the same promotion, persistent | 5, 4 | 395, 185 | 0.15 |
| mediator | turns incentive -> orders around, so the fit adjusts for a confounded mediator | 222, 200 | 178, 0 | 1.2e-89 |
| collider observed | makes sessions, moved by the incentive and demand, a parent of the incentive | 23, 0 | 377, 0 | 1 |
| collider latent | makes sessions, moved by the incentive and a latent that moves supply, a parent | 216, 200 | 184, 0 | 5.9e-96 |
| non-ancestor | says the incentive does not move supply | 44, 0 | 356, 0 | 1 |

In confounder, mediator, collider latent at 0.05.

**The logger check's flag as the stop**, version 2's, on these panels:

| world | the wrong graph | failed, removed | sound, refused | p |
|---|---|---|---|---|
| none | is the true graph | 18, 1 | 182, 10 | 0.66 |
| chase | omits the logger's reading of demand | 24, 12 | 376, 197 | 0.67 |
| sticky | cannot state that the logger keeps half its last incentive | 16, 16 | 184, 184 | 1 |
| confounder | omits a promotion the logger reads and that moves supply | 159, 8 | 241, 14 | 0.71 |
| confounder AR | omits the same promotion, persistent | 5, 3 | 395, 208 | 0.55 |
| mediator | turns incentive -> orders around, so the fit adjusts for a confounded mediator | 222, 11 | 178, 10 | 0.7 |
| collider observed | makes sessions, moved by the incentive and demand, a parent of the incentive | 23, 11 | 377, 196 | 0.73 |
| collider latent | makes sessions, moved by the incentive and a latent that moves supply, a parent | 216, 11 | 184, 9 | 0.56 |
| non-ancestor | says the incentive does not move supply | 44, 4 | 356, 24 | 0.37 |

In no world at 0.05.

## none

The wrong graph is the true graph.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 1.000 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 1.000 | 0.000 | 1.000 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.000 | 1.000 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 1.000 | 0.000 | 1.000 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | — | 0.000 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 1.000 | — | 0.000 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 1.000 (0.999) | 0.0800 |

## chase

The wrong graph omits the logger's reading of demand.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 1.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.000 | 1.000 | 0.060 | 0.000 | 0.005 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 1.000 | 0.000 | 1.000 | 0.060 | 0.000 | 0.005 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | — | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 1.000 | — | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |

## sticky

The wrong graph cannot state that the logger keeps half its last incentive. No graph over the columns states this world's logger, so the oracle arm plans on the same graph as the wrong one.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 1.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 1.000 | 0.000 | 1.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.000 | 1.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 1.000 | 0.000 | 1.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | — | 0.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 1.000 | — | 0.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.999) | 0.0800 |

## confounder

The wrong graph omits a promotion the logger reads and that moves supply.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand, promo | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.998 (0.993) | 0.0800 |
| oracle | stop | demand, promo | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.998 (0.993) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.000 | 1.000 | 0.795 | 0.000 | 0.005 | 0.795 [0.732, 0.849] | 0.993 (0.988) | 0.1415 |
| wrong | stop | demand | 1.000 | 0.000 | 1.000 | 0.795 | 0.000 | 0.005 | 0.795 [0.732, 0.849] | 0.993 (0.988) | 0.1415 |
| by fit | as is | demand, promo | 1.000 | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.998 (0.993) | 0.0800 |
| by fit | stop | demand, promo | 1.000 | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.998 (0.993) | 0.0800 |

## confounder AR

The wrong graph omits the same promotion, persistent.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand, promo | 1.000 | 0.010 | 0.000 | 0.000 | 0.010 | 0.000 | 0.010 [0.001, 0.036] | 0.985 (0.971) | 0.0800 |
| oracle | stop | demand, promo | 0.990 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.985 (0.971) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.010 | 1.000 | 0.000 | 0.015 | 0.000 | 0.015 [0.003, 0.043] | 0.972 (0.966) | 0.1291 |
| wrong | stop | demand | 0.990 | 0.010 | 0.990 | 0.000 | 0.005 | 0.000 | 0.005 [0.000, 0.028] | 0.972 (0.966) | 0.1291 |
| by fit | as is | demand, promo | 1.000 | — | 0.000 | 0.000 | 0.010 | 0.000 | 0.010 [0.001, 0.036] | 0.985 (0.971) | 0.0800 |
| by fit | stop | demand, promo | 1.000 | — | 0.000 | 0.000 | 0.010 | 0.000 | 0.010 [0.001, 0.036] | 0.985 (0.971) | 0.0800 |

## mediator

The wrong graph turns incentive -> orders around, so the fit adjusts for a confounded mediator.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 1.000 | 0.105 | 0.000 | 0.005 | 0.110 [0.070, 0.162] | 0.999 (0.997) | 0.0801 |
| oracle | stop | demand | 1.000 | 0.000 | 1.000 | 0.105 | 0.000 | 0.005 | 0.110 [0.070, 0.162] | 0.999 (0.997) | 0.0801 |
| wrong | as is | demand, orders | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -1.425 (-1.560) | -0.0299 |
| wrong | stop | demand, orders | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | -1.425 (-1.560) | -0.0299 |
| by fit | as is | demand, orders | 1.000 | — | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -1.425 (-1.560) | -0.0299 |
| by fit | stop | demand, orders | 1.000 | — | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -1.425 (-1.560) | -0.0299 |

## collider observed

The wrong graph makes sessions, moved by the incentive and demand, a parent of the incentive.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 1.000 | 0.055 | 0.000 | 0.005 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 1.000 | 0.000 | 1.000 | 0.055 | 0.000 | 0.005 | 0.060 [0.031, 0.102] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand, sessions | 1.000 | 0.000 | 0.000 | 0.055 | 0.000 | 0.000 | 0.055 [0.028, 0.096] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand, sessions | 1.000 | 0.000 | 0.000 | 0.055 | 0.000 | 0.000 | 0.055 [0.028, 0.096] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | — | 0.000 | 0.055 | 0.000 | 0.000 | 0.055 [0.028, 0.096] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 1.000 | — | 0.000 | 0.055 | 0.000 | 0.000 | 0.055 [0.028, 0.096] | 1.000 (0.999) | 0.0800 |

## collider latent

The wrong graph makes sessions, moved by the incentive and a latent that moves supply, a parent.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 1.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.998) | 0.0800 |
| oracle | stop | demand | 1.000 | 0.000 | 1.000 | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 1.000 (0.998) | 0.0800 |
| wrong | as is | demand, sessions | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -0.036 (-0.082) | 0.0001 |
| wrong | stop | demand, sessions | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | -0.036 (-0.082) | 0.0001 |
| by fit | as is | demand, sessions | 1.000 | — | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -0.036 (-0.082) | 0.0001 |
| by fit | stop | demand, sessions | 1.000 | — | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -0.036 (-0.082) | 0.0001 |

## non-ancestor

The wrong graph says the incentive does not move supply.

| arm | reading | adjustment | acted | stopped | evaluated | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.000 | 1.000 | 0.105 | 0.000 | 0.005 | 0.110 [0.070, 0.162] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 1.000 | 0.000 | 1.000 | 0.105 | 0.000 | 0.005 | 0.110 [0.070, 0.162] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.000 | 1.000 | 0.105 | 0.000 | 0.005 | 0.110 [0.070, 0.162] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 1.000 | 0.000 | 1.000 | 0.105 | 0.000 | 0.005 | 0.110 [0.070, 0.162] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | — | 0.000 | 0.105 | 0.000 | 0.000 | 0.105 [0.066, 0.156] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 1.000 | — | 0.000 | 0.105 | 0.000 | 0.000 | 0.105 [0.066, 0.156] | 1.000 (0.999) | 0.0800 |

In each world's table, *stopped* is the stop's rate over the panels whose arm read the draw; the arm chosen by fit reads none. The true channel's push on supply is 0.08 a period in every world. Wall-clock per world is recorded in `track_q_v3.json` and quoted nowhere: it was not measured on a clean machine.
