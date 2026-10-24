# Track Q, version 2 — the analyst's graph is wrong, scored on the tube's claim

200 replicates a world (`causaldyn_bench.graph_errors_v2`, seed 20261022, x64 True), on causal-hybrid-control `d8d87cc`, pre-registered in `results/track_q_v2_preregistration.md` at `e63ba74e4a5ca714f84cbf4f0bf263b2b3b5e8b6`. On each replicate, one panel of 400 units x 12 periods to plan on and a later one to evaluate on. Tolerance 0.8012, 1.1 times the mediator world's 0.95 radius at step 3 under its true plant, 0.7283. A plan leaves when the true plant is outside its tube with probability above 0.05 + 0.0048 = 0.0548 at a step its certificate trusts, read exactly from the Gaussian; it loses when its true regret passes its bound by 0.1 of the stakes. Rate intervals are Clopper-Pearson's.

## Read against the pre-registration

**The gate is not met: in no world does the stop beat a random one** (p from 0.64 to 1; none at
0.05 or at 0.0056). **The size is within two points of 0.05**: 0.058 over 1600 true-graph panels,
[0.047, 0.070]. The tolerance armed the plan leg, as expected: every arm acted on every replicate
read as it is.

Three graph errors fail silently, each with the flag at its size, so the stop removes their
failures no more often than it refuses sound claims:

* **confounder**: the wrong graph's plan leaves its tube on 0.830 of replicates, the true graph's
  on none; the wrong graph's largest exceedance at a trusted step averages 0.059 (95th percentile
  0.066), the true graph's 0.025. The flag fires on 0.055 of its panels.
* **mediator** and **collider latent**: the wrong graph's plan loses on every replicate (its fitted
  channel reads -0.030 and 0.000 against 0.080), the true graph's on none. The flag fires on 0.055
  and 0.040.

In the persistent-confounder world the flag fires on every wrong-graph panel, as in version 1, but
the wrong graph's claims now hold (silent 0.005): its tube is wide at the two steps it trusts
(largest exceedance 0.009), so the stop refuses sound claims and p is 0.78. In none, chase,
sticky, collider observed and non-ancestor the wrong graph leaves exactly as often as the true
graph does: those failures are the tube's own, below.

**The calibration check.** The true graph's tube holds its claim on average and not log by log. In
the six worlds whose tube reads only the plant's noise (none, chase, mediator, the two colliders,
non-ancestor) its exceedance averages 0.050 at each step, as stated; but 0.056 of its trusted steps
[0.049, 0.064] and **0.093 of its plans [0.077, 0.110]** pass 0.05 + m. Pooled over the eight worlds
with an oracle: 0.047 of trusted steps and **0.069 of plans, [0.057, 0.083]**, the two confounder
worlds' tubes, which read the promotion known at the start as noise, at none. By the
pre-registered reading, in none, chase, mediator, collider latent and non-ancestor the plan share's
interval lies above 0.05, so the tube leaves there for a reason that is not the graph, and their
"left" counts are not attributed to the graph error. That costs no attribution above: in those
worlds the wrong graph leaves as often as the true graph or not at all, and the mediator and
collider-latent failures are regret, which the tube does not enter.

Expectations stated in the pre-registration: the leg armed, the confounder world's wrong graph
leaving where its true graph does not, the mediator and collider-latent wrong graphs losing, the
stop at chance in all three, and few failures in the persistent-confounder world: each as
expected. The calibration plan share at about 0.05: not met.

The scored run is the only run on seed 20261022. Before the pre-registration, a dry run of 2
replicates a world on the throwaway seed 1, and one replicate of two worlds on the same seed to
render the calibration table, checked that the code ran; their numbers are not used.

## Amendment, 2026-10-24: a premise that did not hold

The pre-registration derived `m` as the slack of one step and argued that, since one read scale
and one slope serve every step, reading the largest of up to three trusted steps "adds little
beyond the slack of the largest". It does not: in the six noise-only worlds 0.056 of the true
graph's steps pass the threshold and 0.093 of its plans do, so the steps' exceedances do not move
together closely enough for the largest to keep the slack of one. No rule, threshold or number
above is changed: every figure is the pre-registered specification's. A margin for the largest of
three steps would need the joint law of the steps' exceedances over logs, which the pre-registration
did not derive. It would lower the "left" counts; it is not expected to turn a verdict, since the
flag fires at its size on the three attributable failures, and at its size or on every
wrong-graph panel in the other worlds, whatever the tube does. The JSON holds each plan's exceedance at every step,
so any other threshold can be read from it.

## The gate

A flag that does not lower the silent-failure rate is not shipped. Any stop lowers it, so the flag must do better than a stop at the same rate that ignored the failures: remove failures more often than it refuses sound claims (one-sided Fisher exact p). The analyst holds the true graph or the wrong one, so both count, and the wrong one alone where the two are the same. The wrong graph's silent-failure rate, read as it is and with the stop, then what the stop did over both graphs:

| world | the wrong graph | silent | with the stop | failed, removed | sound, refused | p |
|---|---|---|---|---|---|---|
| none | is the true graph | 0.090 | 0.090 | 18, 0 | 182, 15 | 1 |
| chase | omits the logger's reading of demand | 0.120 | 0.000 | 47, 24 | 353, 186 | 0.64 |
| sticky | cannot state that the logger keeps half its last incentive | 0.115 | 0.000 | 23, 23 | 177, 177 | 1 |
| confounder | omits a promotion the logger reads and that moves supply | 0.840 | 0.795 | 168, 9 | 232, 16 | 0.8 |
| confounder AR | omits the same promotion, persistent | 0.005 | 0.000 | 2, 1 | 398, 211 | 0.78 |
| mediator | turns incentive -> orders around, so the fit adjusts for a confounded mediator | 1.000 | 0.945 | 218, 11 | 182, 21 | 0.99 |
| collider observed | makes sessions, moved by the incentive and demand, a parent of the incentive | 0.075 | 0.000 | 30, 15 | 370, 202 | 0.75 |
| collider latent | makes sessions, moved by the incentive and a latent that moves supply, a parent | 1.000 | 0.960 | 220, 8 | 180, 22 | 1 |
| non-ancestor | says the incentive does not move supply | 0.090 | 0.085 | 36, 2 | 364, 40 | 0.91 |

**The flag lowers the silent-failure rate beyond chance: no**, in no world. Over every world the stop removed the failures of 93 replicates and refused a sound claim on 890. At 0.0056, 0.05 over the 9 worlds: no world.

**Its size on these worlds**, the flag rate on the panels the true graph planned on, pooled over the worlds with an oracle: 0.058 over 1600 panels, 95% [0.047, 0.070]; within two points of 0.05: yes.

## The calibration check

On the true graph, the steps its certificate trusts whose exceedance is above 0.0548: how often the tube leaves for a reason that is not the graph. Pooled over the worlds with an oracle: **0.047** of 4306 trusted steps, [0.041, 0.053], and **0.069** of 1600 plans, [0.057, 0.083]. One read scale serves every step of a plan, so the steps' interval, which counts them as independent, is too narrow, and the plans' is the one to read. Per world, with the oracle's mean exceedance at each step over the plans that trust it, and the largest over every plan:

| world | plans acted on | trusted steps | steps above | share of steps | plans that leave | share of plans | mean exceedance, steps 1, 2, 3 | largest |
|---|---|---|---|---|---|---|---|---|
| none | 200 | 600 | 28 | 0.047 [0.031, 0.067] | 17 | 0.085 [0.050, 0.133] | 0.0500, 0.0500, 0.0500 | 0.0581 |
| chase | 200 | 600 | 41 | 0.068 [0.049, 0.092] | 23 | 0.115 [0.074, 0.168] | 0.0502, 0.0503, 0.0503 | 0.0582 |
| sticky (no oracle) | 200 | 600 | 40 | 0.067 [0.048, 0.090] | 22 | 0.110 [0.070, 0.162] | 0.0501, 0.0503, 0.0503 | 0.0593 |
| confounder | 200 | 496 | 0 | 0.000 [0.000, 0.007] | 0 | 0.000 [0.000, 0.018] | 0.0011, 0.0192, 0.0317 | 0.0362 |
| confounder AR | 200 | 210 | 0 | 0.000 [0.000, 0.017] | 0 | 0.000 [0.000, 0.018] | 0.0011, 0.0009, — | 0.0018 |
| mediator | 200 | 600 | 34 | 0.057 [0.040, 0.078] | 18 | 0.090 [0.054, 0.139] | 0.0502, 0.0501, 0.0500 | 0.0585 |
| collider observed | 200 | 600 | 24 | 0.040 [0.026, 0.059] | 15 | 0.075 [0.043, 0.121] | 0.0496, 0.0496, 0.0496 | 0.0582 |
| collider latent | 200 | 600 | 40 | 0.067 [0.048, 0.090] | 20 | 0.100 [0.062, 0.150] | 0.0501, 0.0501, 0.0500 | 0.0619 |
| non-ancestor | 200 | 600 | 34 | 0.057 [0.040, 0.078] | 18 | 0.090 [0.054, 0.139] | 0.0499, 0.0497, 0.0496 | 0.0599 |

## none

The wrong graph is the true graph.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 0.060 | 1.000 | 0.015 | 0.085 | 0.000 | 0.005 | 0.090 [0.054, 0.139] | 0.0509 (0.0558) | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.940 | 3.00 | 0.060 | 0.925 | 0.015 | 0.085 | 0.000 | 0.005 | 0.090 [0.054, 0.139] | 0.0510 (0.0558) | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 3.00 | 0.060 | 1.000 | 0.015 | 0.085 | 0.000 | 0.005 | 0.090 [0.054, 0.139] | 0.0509 (0.0558) | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.940 | 3.00 | 0.060 | 0.925 | 0.015 | 0.085 | 0.000 | 0.005 | 0.090 [0.054, 0.139] | 0.0510 (0.0558) | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 3.00 | 0.065 | 0.000 | — | 0.085 | 0.000 | 0.000 | 0.085 [0.050, 0.133] | 0.0509 (0.0558) | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.935 | 3.00 | 0.065 | 0.000 | — | 0.080 | 0.000 | 0.000 | 0.080 [0.046, 0.127] | 0.0509 (0.0557) | 1.000 (0.999) | 0.0800 |

## chase

The wrong graph omits the logger's reading of demand.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 0.050 | 0.000 | — | 0.115 | 0.000 | 0.000 | 0.115 [0.074, 0.168] | 0.0512 (0.0562) | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.950 | 3.00 | 0.050 | 0.000 | — | 0.115 | 0.000 | 0.000 | 0.115 [0.074, 0.168] | 0.0512 (0.0562) | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 3.00 | 1.000 | 1.000 | 1.000 | 0.115 | 0.000 | 0.005 | 0.120 [0.078, 0.173] | 0.0512 (0.0562) | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.000 | — | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | — | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 3.00 | 0.050 | 0.000 | — | 0.115 | 0.000 | 0.000 | 0.115 [0.074, 0.168] | 0.0512 (0.0562) | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.950 | 3.00 | 0.050 | 0.000 | — | 0.115 | 0.000 | 0.000 | 0.115 [0.074, 0.168] | 0.0512 (0.0562) | 1.000 (0.999) | 0.0800 |

## sticky

The wrong graph cannot state that the logger keeps half its last incentive. No graph over the columns states this world's logger, so the oracle arm plans on the same graph as the wrong one.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 1.000 | 1.000 | 1.000 | 0.110 | 0.000 | 0.005 | 0.115 [0.074, 0.168] | 0.0512 (0.0560) | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.000 | — | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | — | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 3.00 | 1.000 | 1.000 | 1.000 | 0.110 | 0.000 | 0.005 | 0.115 [0.074, 0.168] | 0.0512 (0.0560) | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.000 | — | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | — | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 3.00 | 1.000 | 0.000 | — | 0.110 | 0.000 | 0.000 | 0.110 [0.070, 0.162] | 0.0512 (0.0560) | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.000 | — | 1.000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | — | 1.000 (0.999) | 0.0800 |

## confounder

The wrong graph omits a promotion the logger reads and that moves supply.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand, promo | 1.000 | 2.48 | 0.060 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.0246 (0.0332) | 0.998 (0.994) | 0.0800 |
| oracle | stop | demand, promo | 0.940 | 2.48 | 0.060 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.0245 (0.0331) | 0.998 (0.994) | 0.0800 |
| wrong | as is | demand | 1.000 | 3.00 | 0.055 | 1.000 | 0.065 | 0.830 | 0.000 | 0.015 | 0.840 [0.782, 0.888] | 0.0591 (0.0660) | 0.994 (0.989) | 0.1416 |
| wrong | stop | demand | 0.945 | 3.00 | 0.055 | 0.890 | 0.065 | 0.785 | 0.000 | 0.015 | 0.795 [0.732, 0.849] | 0.0591 (0.0659) | 0.994 (0.989) | 0.1416 |
| by fit | as is | demand, promo | 1.000 | 2.48 | 0.060 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.0246 (0.0332) | 0.998 (0.994) | 0.0800 |
| by fit | stop | demand, promo | 0.940 | 2.47 | 0.060 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | 0.0245 (0.0332) | 0.998 (0.994) | 0.0800 |

## confounder AR

The wrong graph omits the same promotion, persistent.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand, promo | 1.000 | 1.05 | 0.060 | 0.000 | — | 0.000 | 0.005 | 0.000 | 0.005 [0.000, 0.028] | 0.0011 (0.0015) | 0.995 (0.984) | 0.0800 |
| oracle | stop | demand, promo | 0.940 | 1.05 | 0.060 | 0.000 | — | 0.000 | 0.005 | 0.000 | 0.005 [0.000, 0.028] | 0.0011 (0.0015) | 0.995 (0.984) | 0.0800 |
| wrong | as is | demand | 1.000 | 2.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.005 | 0.000 | 0.005 [0.000, 0.028] | 0.0092 (0.0120) | 0.987 (0.971) | 0.1290 |
| wrong | stop | demand | 0.000 | — | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | — | 0.987 (0.971) | 0.1290 |
| by fit | as is | demand, promo | 1.000 | 1.05 | 0.060 | 0.000 | — | 0.000 | 0.005 | 0.000 | 0.005 [0.000, 0.028] | 0.0011 (0.0015) | 0.995 (0.984) | 0.0800 |
| by fit | stop | demand, promo | 0.940 | 1.05 | 0.060 | 0.000 | — | 0.000 | 0.005 | 0.000 | 0.005 [0.000, 0.028] | 0.0011 (0.0015) | 0.995 (0.984) | 0.0800 |

## mediator

The wrong graph turns incentive -> orders around, so the fit adjusts for a confounded mediator.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 0.070 | 1.000 | 0.045 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 0.0511 (0.0556) | 0.999 (0.998) | 0.0799 |
| oracle | stop | demand | 0.930 | 3.00 | 0.070 | 0.890 | 0.045 | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 0.0512 (0.0557) | 0.999 (0.998) | 0.0799 |
| wrong | as is | demand, orders | 1.000 | 2.00 | 0.055 | 0.000 | — | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | 0.0196 (0.0222) | -1.436 (-1.551) | -0.0300 |
| wrong | stop | demand, orders | 0.945 | 2.00 | 0.055 | 0.000 | — | 0.000 | 0.945 | 0.000 | 0.945 [0.904, 0.972] | 0.0197 (0.0223) | -1.436 (-1.551) | -0.0300 |
| by fit | as is | demand, orders | 1.000 | 2.00 | 0.035 | 0.000 | — | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | 0.0196 (0.0222) | -1.436 (-1.551) | -0.0300 |
| by fit | stop | demand, orders | 0.965 | 2.00 | 0.035 | 0.000 | — | 0.000 | 0.965 | 0.000 | 0.965 [0.929, 0.986] | 0.0197 (0.0223) | -1.436 (-1.551) | -0.0300 |

## collider observed

The wrong graph makes sessions, moved by the incentive and demand, a parent of the incentive.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 0.045 | 1.000 | 0.050 | 0.075 | 0.000 | 0.000 | 0.075 [0.043, 0.121] | 0.0505 (0.0551) | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.955 | 3.00 | 0.045 | 0.910 | 0.050 | 0.075 | 0.000 | 0.000 | 0.075 [0.043, 0.121] | 0.0506 (0.0551) | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand, sessions | 1.000 | 3.00 | 1.000 | 0.000 | — | 0.075 | 0.000 | 0.000 | 0.075 [0.043, 0.121] | 0.0505 (0.0551) | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand, sessions | 0.000 | — | 1.000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.018] | — | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 3.00 | 0.045 | 0.000 | — | 0.075 | 0.000 | 0.000 | 0.075 [0.043, 0.121] | 0.0505 (0.0551) | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.955 | 3.00 | 0.045 | 0.000 | — | 0.075 | 0.000 | 0.000 | 0.075 [0.043, 0.121] | 0.0506 (0.0551) | 1.000 (0.999) | 0.0800 |

## collider latent

The wrong graph makes sessions, moved by the incentive and a latent that moves supply, a parent.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 0.050 | 1.000 | 0.070 | 0.100 | 0.000 | 0.000 | 0.100 [0.062, 0.150] | 0.0510 (0.0565) | 1.000 (0.999) | 0.0801 |
| oracle | stop | demand | 0.950 | 3.00 | 0.050 | 0.880 | 0.070 | 0.100 | 0.000 | 0.000 | 0.100 [0.062, 0.150] | 0.0510 (0.0567) | 1.000 (0.999) | 0.0801 |
| wrong | as is | demand, sessions | 1.000 | 3.00 | 0.040 | 0.000 | — | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | 0.0191 (0.0221) | -0.039 (-0.087) | 0.0000 |
| wrong | stop | demand, sessions | 0.960 | 3.00 | 0.040 | 0.000 | — | 0.000 | 0.960 | 0.000 | 0.960 [0.923, 0.983] | 0.0191 (0.0222) | -0.039 (-0.087) | 0.0000 |
| by fit | as is | demand, sessions | 1.000 | 3.00 | 0.020 | 0.000 | — | 0.000 | 1.000 | 0.000 | 1.000 [0.982, 1.000] | 0.0191 (0.0221) | -0.039 (-0.087) | 0.0000 |
| by fit | stop | demand, sessions | 0.980 | 3.00 | 0.020 | 0.000 | — | 0.000 | 0.980 | 0.000 | 0.980 [0.950, 0.995] | 0.0191 (0.0221) | -0.039 (-0.087) | 0.0000 |

## non-ancestor

The wrong graph says the incentive does not move supply.

| arm | reading | adjustment | acted | trusted | flagged | evaluated | flagged later | left | lost | missed | silent | exceedance (95%) | headroom (5%) | channel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | as is | demand | 1.000 | 3.00 | 0.065 | 1.000 | 0.050 | 0.090 | 0.000 | 0.005 | 0.090 [0.054, 0.139] | 0.0507 (0.0560) | 1.000 (0.999) | 0.0800 |
| oracle | stop | demand | 0.935 | 3.00 | 0.065 | 0.890 | 0.050 | 0.085 | 0.000 | 0.000 | 0.085 [0.050, 0.133] | 0.0507 (0.0561) | 1.000 (0.999) | 0.0800 |
| wrong | as is | demand | 1.000 | 3.00 | 0.065 | 1.000 | 0.050 | 0.090 | 0.000 | 0.005 | 0.090 [0.054, 0.139] | 0.0507 (0.0560) | 1.000 (0.999) | 0.0800 |
| wrong | stop | demand | 0.935 | 3.00 | 0.065 | 0.890 | 0.050 | 0.085 | 0.000 | 0.000 | 0.085 [0.050, 0.133] | 0.0507 (0.0561) | 1.000 (0.999) | 0.0800 |
| by fit | as is | demand | 1.000 | 3.00 | 0.065 | 0.000 | — | 0.090 | 0.000 | 0.000 | 0.090 [0.054, 0.139] | 0.0507 (0.0560) | 1.000 (0.999) | 0.0800 |
| by fit | stop | demand | 0.935 | 3.00 | 0.065 | 0.000 | — | 0.085 | 0.000 | 0.000 | 0.085 [0.050, 0.133] | 0.0507 (0.0561) | 1.000 (0.999) | 0.0800 |

In each world's table, *trusted* is the mean number of steps the certificate trusts over the plans acted on, and *exceedance* the mean over them of the largest exceedance at a trusted step, with its 95th percentile. The true channel's push on supply is 0.08 a period in every world. Wall-clock per world is recorded in `track_q_v2.json` and quoted nowhere: it was not measured on a clean machine.
