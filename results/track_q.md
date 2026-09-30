# Track Q — the analyst's graph is wrong

200 replicates a world (`causaldyn_bench.graph_errors`, x64 True): on each, one panel of 400 units x 12 periods to plan on and a later one to evaluate on. Tolerance 0.05; a plan loses when its true regret passes its bound by 0.1 of the stakes. Rate intervals are Clopper-Pearson's.

## The gate

A flag that does not lower the silent-failure rate is not shipped. Any stop lowers it, so the flag must do better than a stop at the same rate that ignored the failures: remove failures more often than it refuses sound claims (one-sided Fisher exact p). The analyst holds the true graph or the wrong one, so both count, and the wrong one alone where the two are the same. The wrong graph's silent-failure rate, read as it is and with the stop, then what the stop did over both graphs:

| world | the wrong graph | silent | with the stop | failed, removed | sound, refused | p |
|---|---|---|---|---|---|---|
| none | is the true graph | 0.000 | 0.000 | 0, 0 | 200, 20 | — |
| chase | omits the logger's reading of demand | 0.000 | 0.000 | 0, 0 | 400, 211 | — |
| sticky | cannot state that the logger keeps half its last incentive | 0.005 | 0.000 | 1, 1 | 199, 199 | 1 |
| confounder | omits a promotion the logger reads and that moves supply | 1.000 | 0.955 | 201, 9 | 199, 4 | 0.13 |
| confounder AR | omits the same promotion, persistent | 0.975 | 0.000 | 213, 195 | 187, 12 | 2e-75 |
| mediator | turns incentive -> orders around, so the fit adjusts for a confounded mediator | 1.000 | 0.970 | 200, 6 | 200, 24 | 1 |
| collider observed | makes sessions, moved by the incentive and demand, a parent of the incentive | 0.000 | 0.000 | 0, 0 | 400, 221 | — |
| collider latent | makes sessions, moved by the incentive and a latent that moves supply, a parent | 1.000 | 0.970 | 200, 6 | 200, 16 | 0.99 |
| non-ancestor | says the incentive does not move supply | 0.000 | 0.000 | 0, 0 | 400, 32 | — |

**The flag lowers the silent-failure rate beyond chance: yes**, in confounder AR. Over every world the stop removed the failures of 217 replicates and refused a sound claim on 739.

**Its size on these worlds**, the flag rate on the panels the true graph planned on, pooled over the worlds with an oracle: 0.046 over 1600 panels, 95% [0.036, 0.057]; within two points of 0.05: yes.

## none

The wrong graph is the true graph.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.065 | 1.000 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.935 | 0.065 | 0.900 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.065 | 1.000 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.935 | 0.065 | 0.900 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 0.045 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.955 | 0.045 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |

## chase

The wrong graph omits the logger's reading of demand.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.055 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.945 | 0.055 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 0.055 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.945 | 0.055 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |

## sticky

The wrong graph cannot state that the logger keeps half its last incentive. No graph over the columns states this world's logger, so the oracle arm plans on the same graph as the wrong one.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.005 | 0.005 [0.000, 0.028] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.005 | 0.005 [0.000, 0.028] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 1.000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.000 | 1.000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |

## confounder

The wrong graph omits a promotion the logger reads and that moves supply.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand, promo | 1.000 | 0.020 | 0.000 | — | 0.005 | 0.000 | 0.000 | 0.005 [0.000, 0.028] | 0.998 (0.991) | 0.0800 |
| oracle | stop | demand, promo | 0.980 | 0.020 | 0.000 | — | 0.005 | 0.000 | 0.000 | 0.005 [0.000, 0.028] | 0.998 (0.991) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.045 | 1.000 | 0.085 | 1.000 | 0.000 | 0.010 | 1.000 [0.982, 1.000] | 0.993 (0.988) | 0.1416 |
| wrong | stop | demand | 0.955 | 0.045 | 0.870 | 0.085 | 0.955 | 0.000 | 0.010 | 0.955 [0.916, 0.979] | 0.993 (0.988) | 0.1416 |
| by fit | as is | demand, promo | 1.000 | 0.035 | 0.000 | — | 0.005 | 0.000 | 0.000 | 0.005 [0.000, 0.028] | 0.998 (0.991) | 0.0800 |
| by fit | stop | demand, promo | 0.965 | 0.035 | 0.000 | — | 0.005 | 0.000 | 0.000 | 0.005 [0.000, 0.028] | 0.998 (0.991) | 0.0800 |

## confounder AR

The wrong graph omits the same promotion, persistent.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand, promo | 1.000 | 0.035 | 0.000 | — | 0.090 | 0.010 | 0.000 | 0.090 [0.054, 0.139] | 0.992 (0.977) | 0.0800 |
| oracle | stop | demand, promo | 0.965 | 0.035 | 0.000 | — | 0.090 | 0.010 | 0.000 | 0.090 [0.054, 0.139] | 0.992 (0.977) | 0.0800 |
| wrong | as is | demand | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 0.005 | 0.025 | 0.975 [0.943, 0.992] | 0.988 (0.976) | 0.1294 |
| wrong | stop | demand | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.988 (0.976) | 0.1294 |
| by fit | as is | demand, promo | 1.000 | 0.040 | 0.000 | — | 0.090 | 0.010 | 0.000 | 0.090 [0.054, 0.139] | 0.992 (0.977) | 0.0800 |
| by fit | stop | demand, promo | 0.960 | 0.040 | 0.000 | — | 0.085 | 0.010 | 0.000 | 0.085 [0.050, 0.133] | 0.992 (0.977) | 0.0800 |

## mediator

The wrong graph turns incentive -> orders around, so the fit adjusts for a confounded mediator.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.070 | 1.000 | 0.055 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.999 (0.998) | 0.0797 |
| oracle | stop | demand | 0.930 | 0.070 | 0.880 | 0.055 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.999 (0.998) | 0.0797 |
| wrong | as is | demand, orders | 1.000 | 0.030 | 0.000 | — | 1.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -1.430 (-1.556) | -0.0300 |
| wrong | stop | demand, orders | 0.970 | 0.030 | 0.000 | — | 0.970 | 0.970 | 0.000 | 0.970 [0.936, 0.989] | -1.430 (-1.556) | -0.0300 |
| by fit | as is | demand, orders | 1.000 | 0.040 | 0.000 | — | 1.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -1.430 (-1.556) | -0.0300 |
| by fit | stop | demand, orders | 0.960 | 0.040 | 0.000 | — | 0.960 | 0.960 | 0.000 | 0.960 [0.923, 0.983] | -1.430 (-1.556) | -0.0300 |

## collider observed

The wrong graph makes sessions, moved by the incentive and demand, a parent of the incentive.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.040 | 1.000 | 0.070 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.960 | 0.040 | 0.895 | 0.070 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand, sessions | 1.000 | 1.000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand, sessions | 0.000 | 1.000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 0.030 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.970 | 0.030 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |

## collider latent

The wrong graph makes sessions, moved by the incentive and a latent that moves supply, a parent.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.040 | 1.000 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.960 | 0.040 | 0.920 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand, sessions | 1.000 | 0.030 | 0.000 | — | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -0.036 (-0.089) | 0.0000 |
| wrong | stop | demand, sessions | 0.970 | 0.030 | 0.000 | — | 0.000 | 0.970 | 0.000 | 0.970 [0.936, 0.989] | -0.036 (-0.089) | 0.0000 |
| by fit | as is | demand, sessions | 1.000 | 0.045 | 0.000 | — | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | -0.036 (-0.089) | 0.0000 |
| by fit | stop | demand, sessions | 0.955 | 0.045 | 0.000 | — | 0.000 | 0.955 | 0.000 | 0.955 [0.916, 0.979] | -0.036 (-0.089) | 0.0000 |

## non-ancestor

The wrong graph says the incentive does not move supply.

| arm | reading | adjustment | acted | flagged | evaluated | flagged later | left | lost | missed | silent | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 0.040 | 1.000 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.960 | 0.040 | 0.920 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 0.040 | 1.000 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.960 | 0.040 | 0.920 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 0.045 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.955 | 0.045 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 1.000 (0.999) | 0.0800 |

The true channel's push on supply is 0.08 a period in every world. Wall-clock per world is recorded in `track_q.json` and quoted nowhere: it was not measured on a clean machine.
