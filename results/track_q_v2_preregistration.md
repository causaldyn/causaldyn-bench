# Track Q, version 2 — pre-registration

Written on 2026-10-24 and committed alone, before any replicate of the scored run was drawn. The
code that will run is `causaldyn_bench.graph_errors_v2` at bench commit
`0ed997289d5af69e8536ce049e5fab66e6c85bfb`, unchanged from there to this document's commit. The results will name this document's commit.

## Why a second version

Track Q (`causaldyn_bench.graph_errors`, `results/track_q.md`) was measured on
causal-hybrid-control 0.8.0. Since 0.17.0 (library commit `f86c2e3`, its ADR 0080) the plan's error
tube carries the plant's own one-step noise. On Track Q's worlds on 0.20 the tube is 0.30 or more
at the first step and 0.72 to 0.79 at the third in the confounder world, above the track's 0.05
tolerance, so no arm acts and the plan leg ("left", "lost") is not armed. The old claim, that the
true path stays within 0.05 over three steps, did not hold either: a realised path stays within
0.05 at the first step in 23–26 % of draws, for the true graph too. The tube the library now
reports makes a different claim: at each step it trusts, the plant's realised path is inside the
tube with probability 0.95. Version 2 scores that claim.

## What does not change

The worlds and their generator, the three arms (the true graph, the wrong one, the adjustment set
chosen by fit), the two readings (as is, and with the logger check's flag as a stop), the flag (the
check at 0.05), the panels (400 units x 12 periods), the decision (supply to 0.3 over 3 periods,
the incentive in [-2, 2] at 0.05 a unit, wait at most 0.5, the same `chc.prescribe` options), the
regret leg, the evaluation leg, the silent failure (any failed claim with nothing refused), the
stop's Fisher test, the gate and the size check. Version 1's module stays and still reproduces
version 1: its new arguments default to version 1's values.

## What changes

### 1. The tolerance, from the generator

The true plant of each world is linear and Gaussian, `s' = A s + b u + w`, `w ~ N(0, Q)`
(`graph_errors.plant`, checked against the panels the generator draws by
`tests/test_graph_errors.py`). From a known start under an open-loop schedule the realised state at
step `k` is Gaussian about the true mean path with covariance

    Sigma_k = sum_{j<k} A^j Q A^j'

and the schedule moves only the mean. The deviation is the Euclidean norm over (supply, wait), the
norm the tube and the tolerance read. The **0.95 predictive radius** `r_k` is the 0.95 quantile of
`||e||`, `e ~ N(0, Sigma_k)` on the two states: the root of `P(l1 X1 + l2 X2 > r^2) = 0.05`, `l1 >=
l2` the eigenvalues of the states' block of `Sigma_k`, `X1`, `X2` independent `chi2_1`. It lies
between `sqrt(3.8415 l1)` and `sqrt(5.9915 l1)`.

In the worlds without a confounder, `A = [[0.94, 0.03], [0, 1.025]]` and `Q = diag(q, 0.01^2)`,
with `q = 0.01^2 + 0.15^2` (the shock and the demand's push) plus the world's own push on supply:
`(0.06 + 0.1)^2 + (0.06 x 1)^2 = 0.0292` in the mediator world (the orders' and the latent's),
`0.1^2` in the collider-latent world (the latent's), none elsewhere. At the plan's last step,

    Sigma_3[supply, supply] = q (1 + a^2 + a^4) + 0.01^2 (c^2 + c^2 (a + d)^2)
    Sigma_3[wait, wait]     = 0.01^2 (1 + d^2 + d^4) = 3.154437890625e-4
    Sigma_3[supply, wait]   = 9.268434375e-6

with `a = 0.94`, `c = 0.03`, `d = 1.025`. In the two confounder worlds the state carries the
promotion `z`, AR(1) at persistence 0.0 or 0.8 with unit variance, pushing supply by 0.2; it is
known at the start, at its mean over the panel's units at the last period, as version 1 reads the
mean path, so it adds variance from the second step on.

| world | `Sigma_3[supply, supply]` | `r_3` |
|---|---|---|
| none, chase, sticky, collider observed, non-ancestor | 0.06021472400625 | 0.48128 |
| collider latent | 0.08685821360625 | 0.57791 |
| confounder AR | 0.11821216400625 | 0.67411 |
| confounder | 0.13555872400625 | 0.72184 |
| mediator | 0.13801371363825 | 0.72835 |

The covariances are exact rationals of the generator's parameters (Maxima); the radii are the root
above, found in Maxima and in the module, which agree to 1e-13.

**The tolerance is 1.1 x 0.728347 = 0.8012**, the mediator world's, the widest. It is one value for
all nine worlds, read off no run. It is the largest so that a plan carrying the true model's own
spread can be trusted for its three steps in every world; at the base worlds' value, 0.5294, the
mediator world's oracle (step-2 tube near 0.62 in the diagnosis) would be trusted one step and its
wrong graph none, and the leg would be unarmed there by the world's noise and not by its graph. The
factor 1.1 is the design's slack: a correct tube also carries the field's error and the noise
read's own error, so it is wider than the plant's radius.

### 2. A plan leaves when its tube's claim fails

For a plan acted on, at each step `k` its certificate trusts (`trustworthy_steps`), the
**exceedance** is

    e_k = P(||x_k - planned_k|| > tube_k)

with `x_k` the true plant's realised (supply, wait) under the plan's actions from the plan's own
start (the confounder at its panel mean, as above), so `x_k ~ N(mu_k, Sigma_k)`, `mu_k` the true
mean path, and `tube_k` the plan's `uncertainty_tube` at step `k`. It is computed exactly, not by
Monte Carlo: along `Sigma_k`'s axes, the two normal tails along the wide axis are integrated over
the narrow coordinate by adaptive quadrature (absolute error 1e-13), and the narrow coordinate past
12 standard deviations is counted as outside (under 1e-32). The test suite checks it against
400 000 draws within four standard errors.

**A plan leaves when `e_k > 0.05 + m` at any trusted step.**

### 3. The margin m = 0.0048

`m` is the estimation slack of a 0.95 tube read off one log: the 95th percentile, over logs, of a
correctly specified tube's exceedance, less 0.05. The log holds 400 units x 11 transitions, `nu =
4400` one-step errors, and nearly all the noise is on supply, so the tube is crossed with
probability `2 (1 - Phi(1.95996 R))`, `R` its radius over the true one. Two reads off the log move
`R`:

* **The noise's scale.** Its variance is read as `sigma^2 chi2_nu / nu`, and the radius scales with
  its root. Alone, in 5 % of logs `R < sqrt(chi2_nu(0.05) / nu) = 0.982442`, where the tube is
  crossed with probability 0.054161: a slack of 0.004161 at every step, exactly. Its relative
  standard deviation is `1 / sqrt(2 nu) = 0.010660`.
* **Supply's persistence `a = 0.94`.** The radius at step `k` scales with `sqrt(sum_{j<k}
  a^(2j))`, so its relative error is `g_k s_a`, `g_k` the log-derivative, 0, 0.4990 and 0.9763 at
  steps 1 to 3, and `s_a = sigma / sqrt(nu v) = 0.005961` the persistence's standard error, `v =
  0.144540` supply's variance given wait in the base worlds' panel, averaged over the 11
  transitions' periods from the panel's law (`Sigma_{t+1} = A Sigma_t A' + 1.5^2 d d' + Q` from
  `0.04 I`, `d = (0.08, -0.04)` the incentive's push).

With `rho_k` the root of the two relative variances' sum, `R` is below `1 - 1.645 rho_k` in 5 % of
logs, where a correct tube is crossed with probability `0.05 + 0.00415`, `0.05 + 0.00432` and `0.05
+ 0.00476` at steps 1 to 3. **m is the largest, 0.00476, rounded to two figures: 0.0048**, one
value at every step and in every world. The leg reads **0.05 + m = 0.0548**.

The tube reads one scale and one slope at every step, so its steps move together and reading any
of three trusted steps adds little beyond the slack of the largest. Ignored, each small: the
degrees of freedom the fit spends (under 1e-5 on `m`); the library's own studentisation of the
read, which widens the tube by about 3e-4 of itself at this `nu` and leans to the tube; the other
worlds' persistence errors, which differ from the base worlds' in the third figure; and the
channel's error, which the tube carries as its field's error. Computed in Maxima (`quantile_chi2`,
`cdf_normal`, the panel recursion in rationals) and in SciPy, which agree to every figure shown.
`m` covers the reads' sampling error and nothing else: a tube that is off for another reason (the
mean a wrong graph fits, a noise form the read misses) is what the leg is for.

### 4. "Lost" and "missed", as before

A plan acted on loses when its true regret on the plant, from its start, passes the certificate's
`regret_bound` by more than 0.1 of the stakes, doing nothing's regret. An evaluation interval from
the later panel misses when it does not cover the schedule's true expected running cost.

## The run

* Seed **20261022**. Replicate `i` of world `w` draws the panel it plans on from NumPy's
  `default_rng([20261022, w, i, 0])` and its later panel from `[20261022, w, i, 1]`, `w` the world's
  index in the order none, chase, sticky, confounder, confounder AR, mediator, collider observed,
  collider latent, non-ancestor.
* **200 replicates a world**, nine worlds, on the CPU, `JAX_ENABLE_X64=1`.
* causal-hybrid-control at commit **`d8d87cc`** (main), imported from its source tree.
* Run once, by `python -m causaldyn_bench.graph_errors_v2 --preregistration <this commit>
  --chc-commit d8d87cc` with four workers.
* Before this document, a dry run of 2 replicates a world on the throwaway seed 1 checked that the
  code runs end to end. Its numbers are not used and are not reported as a result. Two choices
  were made after it, before this document: the calibration check reads plans as well as steps,
  and `m` takes in the persistence's error
  besides the noise scale's (0.0048, from 0.0042 on the scale alone). Both were prompted by the
  dry run, whose plans' steps moved together and whose base worlds' exceedances sat near the
  threshold; their values follow from the derivations above, not from its rates.
* The design was set after a diagnosis on version 1's seed, 20260930: the tube and the exceedances
  of replicates 0–2 of the confounder world and the tube of replicates 0–1 of every world, at
  tolerances 0.05 and 0.9. None of those panels is in the scored run.

## Pass, fail and how they are read

1. **The gate (primary), as before.** Per world, over the true and the wrong graph's replicates, or
   the wrong graph's alone where the two are the same (none, sticky), the stop's table: of the
   replicates whose claims failed, how many it cleared of every failure; of those whose claims held,
   on how many it refused one; the one-sided Fisher exact p that it removes failures more often than
   it refuses sound claims, as a stop at the same rate that ignored the failures would. **The flag
   passes the gate if p <= 0.05 in at least one world.** Reported beside it, not a criterion: the
   worlds where p <= 0.05 / 9 = 0.0056.
2. **The size, as before.** The flag rate on the panels the true graph planned on, pooled over the
   eight worlds with an oracle (all but sticky), with its Clopper–Pearson interval. **It passes if it
   is within 0.02 of 0.05.**
3. **The calibration check.** On the true graph, read as it is, over the trusted steps of the plans
   acted on: the share whose exceedance is above 0.05 + m, per world and pooled over the eight
   worlds with an oracle, with Clopper–Pearson intervals. Beside it, the share of those plans with
   such a step, the true graph's "left" among the plans acted on. One read scale serves every step
   of a plan, so the steps' interval, which counts them as independent, is too narrow; the plans'
   share, over independent replicates, is the one the reading uses. If `m` is the tube's own slack,
   the pooled plan share is about 0.05 or less. Where a world's plan share has its interval's lower
   end above 0.05, its tube leaves for a reason that is not the graph, and its wrong graph's "left"
   counts are reported there but not attributed to the graph error.

## Reported whatever the outcome

* The gate's table per world: the wrong graph's silent-failure rate as it is and with the stop;
  failed and removed; sound and refused; p; the verdict at 0.05 and the reading at 0.0056.
* The size.
* The calibration check, per world and pooled, by step and by plan, with the oracle's mean
  exceedance at each step and the largest.
* Per world, every arm in both readings: the adjustment set, the share acted on, the mean number of
  trusted steps, the flag rates on both panels, the share evaluated, left, lost, missed, silent with
  its interval, the largest exceedance at a trusted step (mean and 95th percentile), the headroom
  and the fitted channel.
* The JSON, per replicate: the tube, the exceedance at each step, the trusted steps, both flags, the
  evaluation and its coverage, the headroom and the regret's excess over its bound, so another
  threshold can be read without a rerun.
* Any departure from this document, as a dated amendment in the results, with the result this
  document specifies beside it wherever it can be computed.

## Expectations, stated before the run

From the diagnosis; not criteria.

* The plan leg is armed: in every world most plans are trusted for at least one step.
* The confounder world's wrong graph leaves (its exceedance at step 3 was 0.057–0.065 on three
  diagnosis replicates) and its true graph does not (0.028–0.032); the flag fires there at its size,
  so the stop does not beat a random one in that world.
* The mediator and collider-latent wrong graphs lose on regret, as in version 1, with the flag at
  its size: the stop does not beat a random one there either.
* In the persistent-confounder world the wrong graph is flagged on every panel. At this tolerance
  its tube is wide at the steps it trusts and its failures may be few; whether the gate passes there
  is open.
* The calibration plan share is about 0.05 pooled, with the worlds whose tube reads only the
  plant's noise near 0.05 and the confounder worlds' true-graph tubes wide, since the tube reads
  the known promotion as noise.

## Limits

* The start is the plan's own, with the promotion known at its panel mean. A unit whose promotion
  is unknown at the start would carry its variance from the first step; that is not scored.
* "The true plant" is the bench's linear-Gaussian plant of each world, which the generator's panels
  are checked against; the worlds are CHC's by construction.
